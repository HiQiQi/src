__author__ = 'QiYE'
from pylearn2.monitor import  Monitor

import logging
from pylearn2.compat import OrderedDict
from pylearn2.datasets.dataset import Dataset
from pylearn2.space import  CompositeSpace
from pylearn2.utils import safe_zip
from pylearn2.utils.iteration import is_stochastic
from pylearn2.utils.data_specs import DataSpecsMapping
log = logging.getLogger(__name__)

class myMonitor(Monitor):

    def setup(self, dataset, cost, batch_size, num_batches=None,
              extra_costs=None, mode='sequential', obj_prereqs=None,
              cost_monitoring_args=None):

        if dataset is None:
            return
        if isinstance(dataset, Dataset):
            dataset = {'': dataset}
        else:
            assert isinstance(dataset, dict)
            assert all(isinstance(key, str) for key in dataset)
            assert all(isinstance(dataset[key], Dataset) for key in dataset)

        if extra_costs is None:
            costs = {}
        else:
            assert isinstance(extra_costs, (OrderedDict, dict))
            costs = extra_costs
        assert '' not in costs
        costs[''] = cost

        if cost_monitoring_args is None:
            cost_monitoring_args = {}

        model = self.model

        # Build a composite data_specs containing the specs for all costs,
        # then the specs of the model
        cost_names = sorted(costs.keys())
        spaces = []
        sources = []
        for c in cost_names:
            c_space, c_source = costs[c].get_data_specs(model)
            spaces.append(c_space)
            sources.append(c_source)

        # Ask the model for the data_specs needed
        m_space, m_source = model.get_monitoring_data_specs()
        spaces.append(m_space)
        sources.append(m_source)

        nested_space = CompositeSpace(spaces)
        nested_sources = tuple(sources)

        # Flatten this data_specs, so we build only one symbolic Theano
        # variable for each of the unique (space, source) pairs.
        mapping = DataSpecsMapping((nested_space, nested_sources))
        space_tuple = mapping.flatten(nested_space, return_tuple=True)
        source_tuple = mapping.flatten(nested_sources, return_tuple=True)
        ipt = tuple(space.make_theano_batch(name='monitor_%s' % source,
                                            batch_size=None)
                    for (space, source) in safe_zip(space_tuple, source_tuple))

        # Build a nested tuple from ipt, to dispatch the appropriate parts
        # of the ipt batch to each cost
        nested_ipt = mapping.nest(ipt)

        # custom_channels = {}
        # for i, cost_name in enumerate(cost_names):
        #     if cost_name == '':
        #         prefix = ''
        #     else:
        #         prefix = cost_name + '_'
        #     cost = costs[cost_name]
        #     cost_ipt = nested_ipt[i]
        #     raw_channels = cost.get_monitoring_channels(model, cost_ipt)
        #     channels = {}
        #     for name in raw_channels:
        #         # We need three things: the value itself (raw_channels[name]),
        #         # the input variables (cost_ipt), and the data_specs for
        #         # these input variables ((spaces[i], sources[i]))
        #         channels[prefix + name] = (raw_channels[name],
        #                                    cost_ipt,
        #                                    (spaces[i], sources[i]))
        #     custom_channels.update(channels)
        #
        # # Use the last inputs from nested_ipt for the model
        # model_channels = model.get_monitoring_channels(nested_ipt[-1])
        # channels = {}
        # for name in model_channels:
        #     # Note: some code used to consider that model_channels[name]
        #     # could be a a (channel, prereqs) pair, this is not supported.
        #     channels[name] = (model_channels[name],
        #                       nested_ipt[-1],
        #                       (spaces[-1], sources[-1]))
        # custom_channels.update(channels)

        if is_stochastic(mode):
            seed = [[2013, 2, 22]]
        else:
            seed = None

        for dataset_name in dataset:
            cur_dataset = dataset[dataset_name]
            self.add_dataset(dataset=cur_dataset,
                             mode=mode,
                             batch_size=batch_size,
                             num_batches=num_batches,
                             seed=seed)
            if dataset_name == '':
                dprefix = ''
            else:
                dprefix = dataset_name + '_'
            # These channel name 'objective' must not vary, since callbacks
            # that respond to the values in the monitor use the name to find
            # it.
            for i, cost_name in enumerate(cost_names):
                cost = costs[cost_name]
                cost_ipt = nested_ipt[i]
                cost_value_list = cost.expr(model, cost_ipt)
                cost_value = reduce(lambda x, y: x + y, cost_value_list)

                if cost_value is not None:
                    if cost_name == '':
                        name = dprefix + 'objective'
                        prereqs = obj_prereqs
                    else:
                        name = dprefix + cost_name
                        prereqs = None

                    cost.get_data_specs(model)[0].validate(cost_ipt)
                    self.add_channel(name=name,
                                     ipt=cost_ipt,
                                     val=cost_value,
                                     data_specs=cost.get_data_specs(model),
                                     dataset=cur_dataset,
                                     prereqs=prereqs)

            # for key in custom_channels:
            #     val, ipt, data_specs = custom_channels[key]
            #     data_specs[0].validate(ipt)
            #     self.add_channel(name=dprefix + key,
            #                      ipt=ipt,
            #                      val=val,
            #                      data_specs=data_specs,
            #                      dataset=cur_dataset)
