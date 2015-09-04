__author__ = 'QiYE'


from theano import function
from theano.gof.op import get_debug_values

from pylearn2.compat import OrderedDict, first_key
from pylearn2.monitor import Monitor
from pylearn2.utils.iteration import  has_uniform_batch_size
from pylearn2.utils import safe_zip

from pylearn2.utils import contains_nan
from pylearn2.utils import contains_inf

from pylearn2.utils.data_specs import DataSpecsMapping

from pylearn2.utils.timing import log_timing
import logging

log = logging.getLogger(__name__)

from pylearn2.training_algorithms.sgd import SGD


class mySGD(SGD):
    def setup(self, model, dataset):
        """
        Compiles the theano functions needed for the train method.

        Parameters
        ----------
        model : a Model instance
        dataset : Dataset
        """
        if self.cost is None:
            self.cost = model.get_default_cost()

        inf_params = [param for param in model.get_params()
                      if contains_inf(param.get_value())]
        if len(inf_params) > 0:
            raise ValueError("These params are Inf: "+str(inf_params))
        if any([contains_nan(param.get_value())
                for param in model.get_params()]):
            nan_params = [param for param in model.get_params()
                          if contains_nan(param.get_value())]
            raise ValueError("These params are NaN: "+str(nan_params))
        self.model = model

        self._synchronize_batch_size(model)
        model._test_batch_size = self.batch_size
        self.monitor = Monitor.get_monitor(model)
        self.monitor._sanity_check()

        # test if force batch size and batch size
        has_force_batch_size = getattr(model, "force_batch_size", False)
        train_dataset_is_uneven = \
            dataset.get_num_examples() % self.batch_size != 0

        has_monitoring_datasets = bool(self.monitoring_dataset)

        if has_monitoring_datasets:
            monitoring_datasets_are_uneven = \
                any(d.get_num_examples() % self.batch_size
                    != 0 for d in self.monitoring_dataset.values())
        else:
            monitoring_datasets_are_uneven = False  # or True it doesn't matter

        if has_force_batch_size and train_dataset_is_uneven and \
           not has_uniform_batch_size(self.train_iteration_mode):

            raise ValueError("Dataset size is not a multiple of batch size."
                             "You should set train_iteration_mode (and "
                             "maybe monitor_iteration_mode) to "
                             "even_sequential, even_shuffled_sequential or "
                             "even_batchwise_shuffled_sequential")

        if has_force_batch_size and has_monitoring_datasets and \
           monitoring_datasets_are_uneven and \
           not has_uniform_batch_size(self.monitor_iteration_mode):

            raise ValueError("Dataset size is not a multiple of batch size."
                             "You should set monitor_iteration_mode to "
                             "even_sequential, even_shuffled_sequential or "
                             "even_batchwise_shuffled_sequential")

        data_specs = self.cost.get_data_specs(self.model)
        mapping = DataSpecsMapping(data_specs)
        space_tuple = mapping.flatten(data_specs[0], return_tuple=True)
        source_tuple = mapping.flatten(data_specs[1], return_tuple=True)

        # Build a flat tuple of Theano Variables, one for each space.
        # We want that so that if the same space/source is specified
        # more than once in data_specs, only one Theano Variable
        # is generated for it, and the corresponding value is passed
        # only once to the compiled Theano function.
        theano_args = []
        for space, source in safe_zip(space_tuple, source_tuple):
            name = '%s[%s]' % (self.__class__.__name__, source)
            arg = space.make_theano_batch(name=name,
                                          batch_size=self.batch_size)
            theano_args.append(arg)
        theano_args = tuple(theano_args)

        # Methods of `self.cost` need args to be passed in a format compatible
        # with data_specs
        nested_args = mapping.nest(theano_args)
        fixed_var_descr = self.cost.get_fixed_var_descr(model, nested_args)
        self.on_load_batch = fixed_var_descr.on_load_batch

        cost_value_list = self.cost.expr(model, nested_args,
                                    ** fixed_var_descr.fixed_vars)
        cost_value = reduce(lambda x, y: x + y, cost_value_list)

        if cost_value is not None and cost_value.name is None:
            # Concatenate the name of all tensors in theano_args !?
            cost_value.name = 'objective'

        learning_rate = self.learning_rate
        params = list(model.get_params())
        assert len(params) > 0
        for i, param in enumerate(params):
            if param.name is None:
                param.name = 'sgd_params[%d]' % i

        grads, updates = self.cost.get_gradients(model, nested_args,
                                                 ** fixed_var_descr.fixed_vars)
        if not isinstance(grads, OrderedDict):
            raise TypeError(str(type(self.cost)) + ".get_gradients returned " +
                            "something with" + str(type(grads)) + "as its " +
                            "first member. Expected OrderedDict.")

        for param in grads:
            assert param in params
        for param in params:
            assert param in grads

        for param in grads:
            if grads[param].name is None and cost_value is not None:
                grads[param].name = ('grad(%(costname)s, %(paramname)s)' %
                                     {'costname': cost_value.name,
                                      'paramname': param.name})
            assert grads[param].dtype == param.dtype

        lr_scalers = model.get_lr_scalers()

        for key in lr_scalers:
            if key not in params:
                raise ValueError("Tried to scale the learning rate on " +\
                        str(key)+" which is not an optimization parameter.")

        log.info('Parameter and initial learning rate summary:')
        for param in params:
            param_name = param.name
            if param_name is None:
                param_name = 'anon_param'
            lr = learning_rate.get_value() * lr_scalers.get(param,1.)
            log.info('\t' + param_name + ': ' + str(lr))

        if self.learning_rule:
            updates.update(self.learning_rule.get_updates(
                learning_rate, grads, lr_scalers))
        else:
            # Use standard SGD updates with fixed learning rate.
            updates.update( dict(safe_zip(params, [param - learning_rate * \
                lr_scalers.get(param, 1.) * grads[param]
                                    for param in params])))

        for param in params:
            if updates[param].name is None:
                updates[param].name = 'sgd_update(' + param.name + ')'
        model.modify_updates(updates)
        for param in params:
            update = updates[param]
            if update.name is None:
                update.name = 'censor(sgd_update(' + param.name + '))'
            for update_val in get_debug_values(update):
                if contains_inf(update_val):
                    raise ValueError("debug value of %s contains infs" %
                            update.name)
                if contains_nan(update_val):
                    raise ValueError("debug value of %s contains nans" %
                            update.name)


        # Set up monitor to model the objective value, learning rate,
        # momentum (if applicable), and extra channels defined by
        # the cost.
        # We have to do that after learning_rule.get_updates has been
        # called, since it may have an effect on
        # learning_rule.add_channels_to_monitor (that is currently the case
        # for AdaDelta and RMSProp).
        self._setup_monitor()

        with log_timing(log, 'Compiling sgd_update'):
            self.sgd_update = function(theano_args,
                                       updates=updates,
                                       name='sgd_update',
                                       on_unused_input='ignore',
                                       mode=self.theano_function_mode)
        self.params = params
    def _setup_monitor(self):
        """
        Set up monitor to model the objective value, learning rate,
        momentum (if applicable), and extra channels defined by
        the cost.

        This method must be called after `learning_rule.get_updates`,
        since it may have an effect on `learning_rule.add_channels_to_monitor`
        (that is currently the case for `learning_rule.RMSProp`).
        """
        if bool(self.monitoring_dataset):
            if (self.monitoring_batch_size is None and
                    self.monitoring_batches is None):
                self.monitoring_batch_size = self.batch_size
                self.monitoring_batches = self.batches_per_iter
            self.monitor.setup_costlist(dataset=self.monitoring_dataset,
                               cost=self.cost,
                               batch_size=self.monitoring_batch_size,
                               num_batches=self.monitoring_batches,
                               extra_costs=self.monitoring_costs,
                               mode=self.monitor_iteration_mode)
            # dataset_name = first_key(self.monitoring_dataset)
            # monitoring_dataset = self.monitoring_dataset[dataset_name]
            #TODO: have Monitor support non-data-dependent channels
            # self.monitor.add_channel(name='learning_rate',
            #                          ipt=None,
            #                          val=self.learning_rate,
            #                          data_specs=(NullSpace(), ''),
            #                          dataset=monitoring_dataset)
            #
            # if self.learning_rule:
            #     self.learning_rule.add_channels_to_monitor(
            #             self.monitor,
            #             monitoring_dataset)