__author__ = 'QiYE'
import logging
import theano
from theano.compat.python2x import OrderedDict
from theano.sandbox.rng_mrg import MRG_RandomStreams
from pylearn2.monitor import get_monitor_doc
from pylearn2.space import CompositeSpace
from pylearn2.utils import is_iterable
from pylearn2.utils import safe_zip
from theano.compat.six.moves import zip as izip
from pylearn2.utils import py_float_types

from functools import wraps
from pylearn2.costs.cost import Cost, NullDataSpecsMixin, SumOfCosts
from pylearn2.utils.exc import reraise_as
import constants
# Only to be used by the deprecation warning wrapper functions

logging.basicConfig()
#logging.getLogger().setLevel(logging.DEBUG)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
from mlp2 import ConvRectifiedLinear, MLP, CompositeLayer, Linear, BadInputSpaceError, RectifiedLinear
from pylearn2.linear.conv2d import *
import pylearn2.utils.serial as serial



class MLPLayer(MLP):
    def __init__(self, model_path=None, latent_space=None, latent_source='latents',y_space = None, y_source=None, **kwargs):
        self.path = model_path
        self.latent_space = latent_space
        self.latent_source = latent_source
        # self._target_space = y_space
        # self._target_source = y_source

        super(MLPLayer, self).__init__(**kwargs)

    # def get_target_source(self):
    #     """
    #     Returns a string, stating the source for the output. By default the
    #     model expects only one output source, which is called 'targets'.
    #     """
    #     if hasattr(self, 'target_source'):
    #         return self.target_source
    #     else:
    #         # return 'targets'
    #         rvals = []
    #         for i in xrange(self.layers[-1].num_layers):
    #             str = 'targets%d'%i
    #             rvals.append(str)
    #         return tuple(rvals)

    def _update_layer_input_spaces(self):
        """
        Tells each layer what its input space should be.

        Notes
        -----
        This usually resets the layer's parameters!
        """
        layers = self.layers
        try:
            layers[0].set_input_space(self.get_input_space())
        except BadInputSpaceError, e:
            raise TypeError("Layer 0 (" + str(layers[0]) + " of type " +
                            str(type(layers[0])) +
                            ") does not support the MLP's "
                            + "specified input space (" +
                            str(self.get_input_space()) +
                            " of type " + str(type(self.get_input_space())) +
                            "). Original exception: " + str(e))
        for i in xrange(1, len(layers)):
            layers[i].set_input_space(layers[i-1].get_output_space())

        if self.path is not None:
            model_load = serial.load(self.path)
            for i in xrange(len(layers)):
                layers[i] = model_load.layers[i]

    def dropout_fprop(self, state_below, latents, default_input_include_prob=0.5,
                      input_include_probs=None, default_input_scale=2.,
                      input_scales=None, per_example=True):

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self._validate_layer_names(list(input_include_probs.keys()))
        self._validate_layer_names(list(input_scales.keys()))

        theano_rng = MRG_RandomStreams(max(self.rng.randint(2 ** 15), 1))

        for layer in self.layers:
            layer_name = layer.layer_name

            if layer_name in input_include_probs:
                include_prob = input_include_probs[layer_name]
            else:
                include_prob = default_input_include_prob

            if layer_name in input_scales:
                scale = input_scales[layer_name]
            else:
                scale = default_input_scale

            state_below = self.apply_dropout(
                state=state_below,
                include_prob=include_prob,
                theano_rng=theano_rng,
                scale=scale,
                mask_value=layer.dropout_input_mask_value,
                input_space=layer.get_input_space(),
                per_example=per_example
            )

            if isinstance(layer, myCompSplitNodesLayer):
                state_below= layer.tree_fprop(state_below, latents)
            elif isinstance(layer, myCompLayer):
                state_below, idx_below = layer.myfprop(state_below)
            else:
                state_below = layer.fprop(state_below)

        return state_below

    def get_layer_monitoring_channels(self, state_below=None,latents=None,
                                    state=None, targets=None):

        rval = OrderedDict()
        state = state_below
        for layer in self.layers:
            # We don't go through all the inner layers recursively
            state_below = state
            if isinstance(layer, myCompSplitNodesLayer):
                state = layer.tree_fprop(state,latents)
            elif isinstance(layer, myCompLayer):
                state = layer.myfprop(state)
            else:
                state = layer.fprop(state)

            args = [state_below, state]
            if layer is self.layers[-1] and targets is not None:
                args.append(targets)

            if layer is self.layers[-1] or not self.silence or isinstance(layer, myCompSplitNodesLayer):
                ch = layer.get_layer_monitoring_channels(*args)
            else:
                ch = OrderedDict()

            if not isinstance(ch, OrderedDict):
                raise TypeError(str((type(ch), layer.layer_name)))
            for key in ch:
                value = ch[key]
                doc = get_monitor_doc(value)
                if doc is None:
                    doc = str(type(layer)) + \
                        ".get_monitoring_channels_from_state did" + \
                        " not provide any further documentation for" + \
                        " this channel."
                doc = 'This channel came from a layer called "' + \
                        layer.layer_name + '" of an MLP.\n' + doc
                value.__doc__ = doc
                rval[layer.layer_name+'_'+key] = value

        return rval

    def fprop(self, state_below, latents):
        idx = T.arange(0,self.batch_size,1)
        for layer in self.layers:
            if isinstance(layer, myCompSplitNodesLayer):
                state_below, idx, latents = layer.tree_fprop(state_below, idx, latents)
            elif isinstance(layer, myCompLayer):
                state_below = layer.myfprop(state_below)
            else:
                state_below = layer.fprop(state_below)


        return state_below,idx,latents

    def get_monitoring_data_specs(self):
        """
        Returns data specs requiring both inputs and targets.

        Returns
        -------
        data_specs: TODO
            The data specifications for both inputs and targets.
        """
        space = CompositeSpace((self.get_input_space(),
                                self.get_target_space(),self.get_latent_space()))
        source = (self.get_input_source(), self.get_target_source(),self.get_latent_source())
        return (space, source)

class myCompSplitNodesLayer(CompositeLayer):

    def __init__(self, layer_name, layers, thresh=None, inputs_to_layers=None):

        P = sharedX(thresh)
        P.name = layer_name + '_P'
        self.P = P

        if inputs_to_layers is  None:
            self.inputs_to_layers = inputs_to_layers

        super(myCompSplitNodesLayer,
              self).__init__(layer_name=layer_name,
                             layers=layers, inputs_to_layers=inputs_to_layers)

        self.__dict__.update(locals())

        del self.self

    def routing_needed(self):
        return self.inputs_to_layers is not None
    def set_input_space(self, space):
        if not isinstance(space, CompositeSpace):
            if self.inputs_to_layers is not None:
                raise ValueError("CompositeLayer received an inputs_to_layers "
                                 "mapping, but does not have a CompositeSpace "
                                 "as its input space, so there is nothing to "
                                 "map. Received " + str(space) + " as input "
                                 "space.")
            else:
                self.layers_to_inputs = OrderedDict()
                for i, layer in enumerate(self.layers):
                    layer.set_input_space(space)
        else:
            if self.num_layers != len(space.components)*2:
                raise ValueError('The num of space componets should meet with the num of composite layers')
            else:
                self.layers_to_inputs = OrderedDict()
                for i, layer in enumerate(self.layers):
                    self.layers_to_inputs[i] = [i/2]
                    cur_space = space.restrict(self.layers_to_inputs[i])
                    layer.set_input_space(cur_space)


        self.input_space = space
        self.output_space = CompositeSpace(tuple(layer.get_output_space()
                                                 for layer in self.layers))
        self._target_space = CompositeSpace(tuple(layer.get_target_space()
                                                  for layer in self.layers))
    def tree_fprop(self, state_below, idx, latents):

        rvals = []
        rlats= []
        ridx = []
        if isinstance(self.input_space, CompositeSpace):
            for i in xrange(len(self.layers)/2):

                cur_state_below = state_below[i]
                current_latent = latents[i]
                current_idx = idx[i]
                indict = latents[i] - self.P[i]

                left_idx = (indict[:,0]>0).nonzero()
                rlats.append(current_latent[left_idx])
                ridx.append(current_idx[left_idx])

                left_child_input =cur_state_below[left_idx]


                tmp = self.layers[i*2].fprop(left_child_input)
                rvals.append(tmp)

                right_idx = (indict[:,0]<=0).nonzero()
                rlats.append(current_latent[right_idx])
                ridx.append(current_idx[right_idx])


                right_child_input = cur_state_below[right_idx]
                tmp = self.layers[i*2+1].fprop(right_child_input)
                rvals.append(tmp)
        else:

                cur_state_below = state_below
                indict = latents - self.P[0]

                left_idx = (indict[:,0]>0).nonzero()
                rlats.append(latents[left_idx])
                ridx.append(idx[left_idx])

                left_child_input =cur_state_below[left_idx]
                tmp = self.layers[0].fprop(left_child_input)
                rvals.append(tmp)

                right_idx = (indict[:,0]<=0).nonzero()
                ridx.append(idx[right_idx])
                rlats.append(latents[right_idx])

                right_child_input = cur_state_below[right_idx]
                tmp = self.layers[1].fprop(right_child_input)
                rvals.append(tmp)

        return tuple(rvals), tuple(ridx), tuple(rlats)

    def get_lr_scalers(self):

        rval = OrderedDict()

        for param in self.get_params():
            rval[param] = 1.0

        return rval

    def get_weight_decay(self, coeff):
        method_name = 'get_weight_decay'
        if isinstance(coeff, py_float_types):
           decay_list = [getattr(layer, method_name)(coeff)
                          for layer in self.layers]
        elif is_iterable(coeff):
            assert all(layer_coeff >= 0 for layer_coeff in coeff)
            decay_list = [getattr(layer, method_name)(layer_coeff) for
                          layer, layer_coeff in safe_zip(self.layers, coeff)
                          if layer_coeff > 0]
        else:
            raise TypeError("CompositeLayer's " + method_name + " received "
                            "coefficients of type " + str(type(coeff)) + " "
                            "but must be provided with a float or list/tuple")

        return decay_list


class myCompLayer(CompositeLayer):
    def set_input_space(self, space):
        if not isinstance(space, CompositeSpace):
            raise ValueError("CompositeLayer received an inputs_to_layers "
                             "mapping, but does not have a CompositeSpace "
                             "as its input space, so there is nothing to "
                             "map. Received " + str(space) + " as input "
                             "space.")

        else:
            if self.num_layers != len(space.components):
                raise ValueError('The num of space componets should meet with the num of composite layers')
            else:
                self.layers_to_inputs = OrderedDict()
                for i, layer in enumerate(self.layers):
                    self.layers_to_inputs[i] = [i]
                    cur_space = space.restrict(self.layers_to_inputs[i])
                    layer.set_input_space(cur_space)


        self.input_space = space
        self.output_space = CompositeSpace(tuple(layer.get_output_space()
                                                 for layer in self.layers))
        self._target_space = self.layers[0].get_target_space()

        ###############################################################################


    def myfprop(self, state_below):
        rvals = []
        for i, layer in enumerate(self.layers):
            rvals.append(layer.fprop(state_below[i]))
        return tuple(rvals)

    def cost(self, Y, Y_hat,idx):

        cost_list = []
        n_layers = len(self.layers)
        for i in xrange(n_layers):
            current_idx = idx[i]
            current_Y = Y[current_idx]
            cost_list.append(self.layers[i].cost(current_Y, Y_hat[i]))
            # cost_list.append(self.layers[i].cost(Y_hat[i], Y_hat[i]))
        return cost_list


    def get_weight_decay(self, coeff):
        method_name = 'get_weight_decay'
        if isinstance(coeff, py_float_types):
           decay_list = [getattr(layer, method_name)(coeff)
                          for layer in self.layers]
        elif is_iterable(coeff):
            assert all(layer_coeff >= 0 for layer_coeff in coeff)
            decay_list = [getattr(layer, method_name)(layer_coeff) for
                          layer, layer_coeff in safe_zip(self.layers, coeff)
                          if layer_coeff > 0]
        else:
            raise TypeError("CompositeLayer's " + method_name + " received "
                            "coefficients of type " + str(type(coeff)) + " "
                            "but must be provided with a float or list/tuple")

        return decay_list

class myRectLinear(RectifiedLinear):

    def fprop(self, state_below):
        p = self._linear_part(state_below)
        p = T.switch(p > 0., p, self.left_slope * p)
        p = p/T.max(p, axis=1, keepdims=True)
        return p

class myLinear(Linear):

    def cost(self, Y, Y_hat):
        return self.cost_from_cost_matrix(self.cost_matrix(Y, Y_hat))

    def cost_from_cost_matrix(self, cost_matrix):
        return cost_matrix.sum(axis=1).mean()

    def cost_matrix(self, Y, Y_hat):
        if(self.use_abs_loss):
            return T.abs_(Y - Y_hat)
        else:
            diff = T.sqr(Y - Y_hat)
            return T.sum(T.reshape(diff,(diff.shape[0],constants.NUM_JNTS,constants.OUTDIM )), axis=-1)


class myWeightDecay(NullDataSpecsMixin, Cost):
    """L2 regularization cost for MLP.

    coeff * sum(sqr(weights)) for each set of weights.

    Parameters
    ----------
    coeffs : list
        One element per layer, specifying the coefficient to multiply
        with the cost defined by the squared L2 norm of the weights for
        each layer.

        Each element may in turn be a list, e.g., for CompositeLayers.
    """

    def __init__(self, coeffs):
        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):

        self.get_data_specs(model)[0].validate(data)

        n_leafnodes = len(model.layers[-1].layers)
        n_model_layer = len(model.layers)
        decay0 = model.layers[0].get_weight_decay(self.coeffs[0]/n_leafnodes)
        rval = [decay0]*n_leafnodes

        for i in xrange(1,n_model_layer,1):
            currLayer = model.layers[i]
            coeff = self.coeffs[i]
            if isinstance(currLayer, myCompSplitNodesLayer):
                n_curnodes = len(currLayer.layers)
                step = int(n_leafnodes / n_curnodes)
                decay_list = currLayer.get_weight_decay(coeff)
                for k in xrange(n_leafnodes):
                    rval[k] += decay_list[k/step]
            else:
                decay_list = currLayer.get_weight_decay(coeff)
                for k in xrange(n_leafnodes):
                    rval[k] += decay_list[k]

        return rval

    def get_gradients(self, model, data, ** kwargs):


        try:
            cost = self.expr(model=model, data=data, **kwargs)
        except TypeError:
            # If anybody knows how to add type(self) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            message = "Error while calling " + str(type(self)) + ".expr"
            reraise_as(TypeError(message))

        if cost is None:
            raise NotImplementedError(str(type(self)) +
                                      " represents an intractable cost and "
                                      "does not provide a gradient "
                                      "approximation scheme.")

        n_leafnodes = len(cost)
        n_model_layer = len(model.layers)
        params = []
        for i in xrange(n_leafnodes):
            params_branch = []
            layer_params = model.layers[0].get_params()
            for param in layer_params:
                params_branch.append(param)
            params.append(params_branch)

        for i in xrange(1,n_model_layer - constants.NUM_REGLAYER,1):
            CompsiteLayer =  model.layers[i]
            n_curnodes = len(CompsiteLayer.layers)
            step = int(n_leafnodes / n_curnodes)
            for k in xrange(n_leafnodes):
                layer_params = CompsiteLayer.layers[k/step].get_params()
                for param in layer_params:
                    params[k].append(param)

        for i in xrange(-constants.NUM_REGLAYER,0,1):
            CompsiteLayer =  model.layers[i]
            for k in xrange(n_leafnodes):
                layer_params = CompsiteLayer.layers[k].get_params()
                for param in layer_params:
                    params[k].append(param)

        grads =[]
        for i in xrange(n_leafnodes):
            grads.append(theano.tensor.grad(cost[i], params[i], disconnected_inputs='ignore'))


        flat_grads=[]
        flat_params = []
        for j in xrange(n_model_layer - constants.NUM_REGLAYER):
            weight = params[0][2*j]
            bais = params[0][2*j+1]

            tmp_grad_weigth = grads[0][2*j]
            tmp_grad_bais = grads[0][2*j+1]

            for i in  xrange(1,n_leafnodes,1):
                if weight.name == params[i][2*j].name:
                    weight += params[i][2*j]
                    weight.name = params[i][2*j].name
                    bais += params[i][2*j+1]
                    bais.name = params[i][2*j+1].name


                    tmp_grad_weigth += grads[i][2*j]
                    tmp_grad_bais += grads[i][2*j+1]
                else:
                    flat_params.append(weight)
                    flat_params.append(bais)
                    flat_grads.append(tmp_grad_weigth)
                    flat_grads.append(tmp_grad_bais)

                    weight = params[i][2*j]
                    bais = params[i][2*j+1]

                    tmp_grad_weigth = grads[i][2*j]
                    tmp_grad_bais = grads[i][2*j+1]


            flat_params.append(weight)
            flat_params.append(bais)
            flat_grads.append(tmp_grad_weigth)
            flat_grads.append(tmp_grad_bais)


        for j in xrange(-constants.NUM_REGLAYER,0,1):
             for i in xrange(0, n_leafnodes,1):
                flat_params.append(params[i][2*j])
                flat_params.append(params[i][2*j+1])
                flat_grads.append(grads[i][2*j])
                flat_grads.append(grads[i][2*j+1])

        params = model.get_params()
        if len(flat_params) != len(params):
                   raise ValueError("the length of the flat_params of tree cnn "
                                    "does not meet the list of model params" )
        else:
            for flat_i, p_i in zip(flat_params,params):
                flat_i.name = p_i.name
        gradients = OrderedDict(izip(params, flat_grads))

        updates = OrderedDict()

        return gradients, updates

    @wraps(Cost.is_stochastic)
    def is_stochastic(self):
        return False

class myDataSpecsMixin(object):
    """
    Use multiple inheritance with both this object and Cost in order to
    obtain the default data specification.

    Due to method resolution order, you want Cost to appear after
    DefaultDataSpecsMixin in the superclass list.
    """

    def get_data_specs(self, model):
        """
        Provides a default data specification.

        The cost requests input features from the model's input space and
        input source. `self` must contain a bool field called `supervised`.
        If this field is True, the cost requests targets as well.

        Parameters
        ----------
        model : pylearn2.models.Model
            TODO WRITEME
        """
        if self.supervised:
            # b=model.get_input_space()
            # a = model.get_latent_space()
            # space = CompositeSpace([model.get_input_space(),
            #                         CompositeSpace([model.get_target_space(),model.get_latent_space()])])
            # sources = (model.get_input_source(), (model.get_target_source(),model.get_latent_source()))
            # mapping = DataSpecsMapping((space, sources))
            # flat_source = mapping.flatten(sources)
            # # flat_source == ('features', 'features', 'targets')
            # flat_space = mapping.flatten(space)
            # return (flat_space, flat_source)
            space = CompositeSpace([model.get_input_space(),
                                    model.get_target_space(),model.get_latent_space()])
            sources = (model.get_input_source(), model.get_target_source(),model.get_latent_source())
            return space, sources
        else:
            return (model.get_input_space(), model.get_input_source())

class myDropoutCost(myDataSpecsMixin, Cost):


    supervised = True

    def __init__(self, default_input_include_prob=.5, input_include_probs=None,
            default_input_scale=2., input_scales=None, weight=5.0, per_example=True, cost_type = 0):

        if input_include_probs is None:
            input_include_probs = {}

        if input_scales is None:
            input_scales = {}

        self.__dict__.update(locals())
        del self.self

    def expr(self, model, data, ** kwargs):
        """
        .. todo::
        .. todo::

            WRITEME
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y, latent) = data
        Y_hat = model.dropout_fprop(
            state_below= X,
            latents = latent,
            default_input_include_prob=self.default_input_include_prob,
            input_include_probs=self.input_include_probs,
            default_input_scale=self.default_input_scale,
            input_scales=self.input_scales,
            per_example=self.per_example
        )
        return model.layers[-1].cost(Y, Y_hat)
        # if self.cost_type == 0:
        #     return tuple(model.cost(Y, Y_hat), self.weight*reg_cost)
        # else:
        #     return model.cost(Y, Y_hat)
        # return reg_cost

    def get_gradients(self, model, data, ** kwargs):


        try:
            cost = self.expr(model=model, data=data, **kwargs)
        except TypeError:
            # If anybody knows how to add type(self) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            message = "Error while calling " + str(type(self)) + ".expr"
            reraise_as(TypeError(message))

        if cost is None:
            raise NotImplementedError(str(type(self)) +
                                      " represents an intractable cost and "
                                      "does not provide a gradient "
                                      "approximation scheme.")

        n_leafnodes = len(cost)
        n_model_layer = len(model.layers)
        params = []
        for i in xrange(n_leafnodes):
            params_branch = []
            layer_params = model.layers[0].get_params()
            for param in layer_params:
                params_branch.append(param)
            params.append(params_branch)

        for i in xrange(1,n_model_layer - constants.NUM_REGLAYER,1):
            CompsiteLayer =  model.layers[i]
            n_curnodes = len(CompsiteLayer.layers)
            step = int(n_leafnodes / n_curnodes)
            for k in xrange(n_leafnodes):
                layer_params = CompsiteLayer.layers[k/step].get_params()
                for param in layer_params:
                    params[k].append(param)

        for i in xrange(-constants.NUM_REGLAYER,0,1):
            CompsiteLayer =  model.layers[i]
            for k in xrange(n_leafnodes):
                layer_params = CompsiteLayer.layers[k].get_params()
                for param in layer_params:
                    params[k].append(param)

        grads =[]
        for i in xrange(n_leafnodes):
            grads.append(theano.tensor.grad(cost[i], params[i], disconnected_inputs='ignore'))


        flat_grads=[]
        flat_params = []
        for j in xrange(n_model_layer - constants.NUM_REGLAYER):
            weight = params[0][2*j]
            bais = params[0][2*j+1]

            tmp_grad_weigth = grads[0][2*j]
            tmp_grad_bais = grads[0][2*j+1]

            for i in  xrange(1,n_leafnodes,1):
                if weight.name == params[i][2*j].name:
                    weight += params[i][2*j]
                    weight.name = params[i][2*j].name
                    bais += params[i][2*j+1]
                    bais.name = params[i][2*j+1].name

                    tmp_grad_weigth += grads[i][2*j]
                    tmp_grad_bais += grads[i][2*j+1]
                else:
                    flat_params.append(weight)
                    flat_params.append(bais)

                    flat_grads.append(tmp_grad_weigth)
                    flat_grads.append(tmp_grad_bais)

                    weight = params[i][2*j]
                    bais = params[i][2*j+1]

                    tmp_grad_weigth = grads[i][2*j]
                    tmp_grad_bais = grads[i][2*j+1]


            flat_params.append(weight)
            flat_params.append(bais)
            flat_grads.append(tmp_grad_weigth)
            flat_grads.append(tmp_grad_bais)


        for j in xrange(-constants.NUM_REGLAYER,0,1):
             for i in xrange(0, n_leafnodes,1):
                flat_params.append(params[i][2*j])
                flat_params.append(params[i][2*j+1])
                flat_grads.append(grads[i][2*j])
                flat_grads.append(grads[i][2*j+1])

        params_model = model.get_params()
        if len(flat_params) != len(params_model):
                   raise ValueError("the length of the flat_params of tree cnn "
                                    "does not meet the list of model params" )
        else:
            for flat_i, p_i in zip(flat_params,params_model):
                flat_i.name = p_i.name
        gradients = OrderedDict(izip(params_model, flat_grads))

        updates = OrderedDict()

        return gradients, updates

class myCost(myDataSpecsMixin, Cost):

    supervised = True

    def expr(self, model, data, ** kwargs):
        """
        .. todo::
        .. todo::

            WRITEME
        """
        space, sources = self.get_data_specs(model)
        space.validate(data)
        (X, Y, latent) = data
        Y_hat, idx, latents = model.fprop(
            state_below= X,
            latents = latent
        )

        return model.layers[-1].cost(Y, Y_hat, idx)


    def get_gradients(self, model, data, ** kwargs):


        try:
            cost = self.expr(model=model, data=data, **kwargs)
        except TypeError:
            # If anybody knows how to add type(self) to the exception message
            # but still preserve the stack trace, please do so
            # The current code does neither
            message = "Error while calling " + str(type(self)) + ".expr"
            reraise_as(TypeError(message))

        if cost is None:
            raise NotImplementedError(str(type(self)) +
                                      " represents an intractable cost and "
                                      "does not provide a gradient "
                                      "approximation scheme.")

        n_leafnodes = len(cost)
        n_model_layer = len(model.layers)
        params = []
        for i in xrange(n_leafnodes):
            params_branch = []
            layer_params = model.layers[0].get_params()
            for param in layer_params:
                params_branch.append(param)
            params.append(params_branch)

        for i in xrange(1,n_model_layer - constants.NUM_REGLAYER,1):
            CompsiteLayer =  model.layers[i]
            n_curnodes = len(CompsiteLayer.layers)
            step = int(n_leafnodes / n_curnodes)
            for k in xrange(n_leafnodes):
                layer_params = CompsiteLayer.layers[k/step].get_params()
                for param in layer_params:
                    params[k].append(param)

        for i in xrange(-constants.NUM_REGLAYER,0,1):
            CompsiteLayer =  model.layers[i]
            for k in xrange(n_leafnodes):
                layer_params = CompsiteLayer.layers[k].get_params()
                for param in layer_params:
                    params[k].append(param)

        grads =[]
        for i in xrange(n_leafnodes):
            grads.append(theano.tensor.grad(cost[i], params[i], disconnected_inputs='ignore'))


        flat_grads=[]
        flat_params = []
        for j in xrange(n_model_layer - constants.NUM_REGLAYER):
            weight = params[0][2*j]
            bais = params[0][2*j+1]

            tmp_grad_weigth = grads[0][2*j]
            tmp_grad_bais = grads[0][2*j+1]

            for i in  xrange(1,n_leafnodes,1):
                if weight.name == params[i][2*j].name:
                    weight += params[i][2*j]
                    weight.name = params[i][2*j].name

                    bais += params[i][2*j+1]
                    bais.name = params[i][2*j+1].name

                    tmp_grad_weigth += grads[i][2*j]
                    tmp_grad_bais += grads[i][2*j+1]
                else:
                    flat_params.append(weight)
                    flat_params.append(bais)

                    flat_grads.append(tmp_grad_weigth)
                    flat_grads.append(tmp_grad_bais)

                    weight = params[i][2*j]
                    bais = params[i][2*j+1]

                    tmp_grad_weigth = grads[i][2*j]
                    tmp_grad_bais = grads[i][2*j+1]


            flat_params.append(weight)
            flat_params.append(bais)
            flat_grads.append(tmp_grad_weigth)
            flat_grads.append(tmp_grad_bais)


        for j in xrange(-constants.NUM_REGLAYER,0,1):
             for i in xrange(0, n_leafnodes,1):
                flat_params.append(params[i][2*j])
                flat_params.append(params[i][2*j+1])
                flat_grads.append(grads[i][2*j])
                flat_grads.append(grads[i][2*j+1])

        params_model = model.get_params()
        if len(flat_params) != len(params_model):
                   raise ValueError("the length of the flat_params of tree cnn "
                                    "does not meet the list of model params" )
        else:
            for flat_i, p_i in zip(flat_params,params_model):
                flat_i.name = p_i.name
        gradients = OrderedDict(izip(params_model, flat_grads))

        updates = OrderedDict()

        return gradients, updates
class mySumOfCosts(SumOfCosts):
    def expr(self, model, data, ** kwargs):
        """
        Returns the sum of the costs the SumOfCosts instance was given at
        initialization.

        Parameters
        ----------
        model : pylearn2.models.model.Model
            the model for which we want to calculate the sum of costs
        data : flat tuple of tensor_like variables.
            data has to follow the format defined by self.get_data_specs(),
            but this format will always be a flat tuple.
        """
        print 'sumofcost exp'
        self.get_data_specs(model)[0].validate(data)
        composite_specs, mapping = self.get_composite_specs_and_mapping(model)
        nested_data = mapping.nest(data)
        costs = []
        for cost, cost_data in safe_zip(self.costs, nested_data):
            costs.append(cost.expr(model, cost_data, **kwargs))
        assert len(costs) > 0

        if any([cost is None for cost in costs]):
            sum_of_costs = None
        else:
            assert len(costs) > 0
            sum_of_costs = [0]*len(costs[0])
            for coeff, cost in safe_zip(self.coeffs, costs):
                for i in xrange(len(cost)):
                    sum_of_costs[i] += cost[i]*coeff

        return sum_of_costs
