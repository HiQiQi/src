__author__ = 'QiYE'
# Local imports
import pylearn2.config.yaml_parse

import jobman
from jobman.tools import expand, flatten
from jobman.tools import DD

import numpy
import constants
import theano
theano.config.optimizer='None'
theano.config.exception_verbosity='high'
class ydict(dict):
    '''
    YAML-friendly subclass of dictionary.

    The special key "__builder__" is interpreted as the name of an object
    constructor.

    For instance, building a ydict from the following dictionary:

        {
            '__builder__': 'pylearn2.training_algorithms.sgd.EpochCounter',
            'max_epochs': 2
        }

    Will be displayed like:

        !obj:pylearn2.training_algorithms.sgd.EpochCounter {'max_epochs': 2}
    '''
    def __str__(self):
        args_dict = dict(self)
        builder = args_dict.pop('__builder__', '')
        ret_list = []
        if builder:
            ret_list.append('!obj:%s {' % builder)
        else:
            ret_list.append('{')

        for key, val in args_dict.iteritems():
            # This will call str() on keys and values, not repr(), so unicode
            # objects will have the form 'blah', not "u'blah'".
            ret_list.append('%s: %s,' % (key, val))

        ret_list.append('}')
        return '\n'.join(ret_list)


def train_experiment(state, channel):
    """
    Train a model specified in state, and extract required results.

    This function builds a YAML string from ``state.yaml_template``, taking
    the values of hyper-parameters from ``state.hyper_parameters``, creates
    the corresponding object and trains it (like train.py), then run the
    function in ``state.extract_results`` on it, and store the returned values
    into ``state.results``.

    To know how to use this function, you can check the example in tester.py
    (in the same directory).
    """
    yaml_template = state.yaml_template

    # Convert nested DD into nested ydict.
    hyper_parameters = expand(flatten(state.hyper_parameters), dict_type=ydict)

    # This will be the complete yaml string that should be executed
    final_yaml_str = yaml_template % hyper_parameters

    # Instantiate an object from YAML string
    train_obj = pylearn2.config.yaml_parse.load(final_yaml_str)

    try:
        iter(train_obj)
        iterable = True
    except TypeError:
        iterable = False
    if iterable:
        raise NotImplementedError(
                ('Current implementation does not support running multiple '
                 'models in one yaml string.  Please change the yaml template '
                 'and parameters to contain only one single model.'))
    else:
        # print "Executing the model."
        train_obj.main_loop()
        # This line will call a function defined by the user and pass train_obj
        # to it.
        state.results = jobman.tools.resolve(state.extract_results)(train_obj)
        return channel.COMPLETE

if __name__ == '__main__':

    state = DD()

    state.yaml_template = '''
    !obj:pylearn2.train.Train {
        dataset:  !obj:Database_hand.Database_latent {
            dataset_path: &train %(trainfile)s,
            N: %(N)i,
            n: %(n)i,
            which_set: 'train',
            which_label: 'uvd'
        },

        "model": !obj:myLayer_tree_faster.MLPLayer {
            "batch_size": %(batch_size)d,
            "latent_space":!obj:pylearn2.space.VectorSpace {
            dim: 1
            },
            "input_space": !obj:pylearn2.space.Conv2DSpace {
            shape: [96, 96],
            num_channels: 1
            },
            "silence": False,
            "layers": [
                    !obj:mlp2.ConvRectifiedLinear {
                        "layer_name": "wrapper1",
                        "output_channels": %(c1)d,
                         irange: %(irange_c1)f,
                         kernel_shape: [%(kernel_c1)d, %(kernel_c1)d,],
                         pool_shape: [%(pool_c1)d, %(pool_c1)d,],
                         pool_stride: [%(pool_c1)d, %(pool_c1)d,],
                    },
                    !obj:myLayer_tree_faster.myCompSplitNodesLayer {
                        "layer_name": "wrapper20",
                        "thresh":[64],
                        # "inputs_to_layers": { 0:[], },
                        "layers": [
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h1",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h2",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                        ],
                    },
                    !obj:myLayer_tree_faster.myCompSplitNodesLayer {
                        "layer_name": "wrapper21",
                        "thresh":[32,96],
                        # "inputs_to_layers": { 0:[], },
                        "layers": [
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h1",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h2",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h3",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h4",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                        ],
                    },
                    !obj:myLayer_tree_faster.myCompSplitNodesLayer {
                        "layer_name": "wrapper22",
                        "thresh":[16,48,80,112],
                        "layers": [
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h1",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h2",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h3",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h4",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h5",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h6",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h7",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            },
                            !obj:mlp2.ConvRectifiedLinear {
                                "layer_name": "h8",
                                 output_channels: %(c2)i,
                                 irange: %(irange_c2)f,
                                 kernel_shape: [%(kernel_c2)d, %(kernel_c2)d],
                                 pool_shape: [%(pool_c2)d, %(pool_c2)d],
                                 pool_stride: [%(pool_c2)d, %(pool_c2)d],
                            }
                        ],
                    },
                    !obj:myLayer_tree_faster.myCompLayer {
                        "layer_name": "wrapper3",
                        # "inputs_to_layers": { 0:[], },
                        "layers": [
                            !obj:myLayer_tree_faster.myRectLinear {
                                "layer_name": "h1",
                                 dim: %(hd1)d,
                                 irange: %(irange_hd1)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear  {
                                "layer_name": "h2",
                                 dim: %(hd1)d,
                                 irange: %(irange_hd1)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear {
                                "layer_name": "h3",
                                 dim: %(hd1)d,
                                 irange: %(irange_hd1)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear  {
                                "layer_name": "h4",
                                 dim: %(hd1)d,
                                 irange: %(irange_hd1)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear {
                                "layer_name": "h5",
                                 dim: %(hd1)d,
                                 irange: %(irange_hd1)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear  {
                                "layer_name": "h6",
                                 dim: %(hd1)d,
                                 irange: %(irange_hd1)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear {
                                "layer_name": "h7",
                                 dim: %(hd1)d,
                                 irange: %(irange_hd1)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear  {
                                "layer_name": "h8",
                                 dim: %(hd1)d,
                                 irange: %(irange_hd1)f
                            }
                        ],
                    },
                    !obj:myLayer_tree_faster.myCompLayer {
                        "layer_name": "wrapper4",
                        "layers": [
                            !obj:myLayer_tree_faster.myRectLinear {
                                "layer_name": "h1",
                                 dim: %(hd2)d,
                                 irange: %(irange_hd2)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear  {
                                "layer_name": "h2",
                                 dim: %(hd2)d,
                                 irange: %(irange_hd2)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear {
                                "layer_name": "h3",
                                 dim: %(hd2)d,
                                 irange: %(irange_hd2)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear  {
                                "layer_name": "h4",
                                 dim: %(hd2)d,
                                 irange: %(irange_hd2)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear {
                                "layer_name": "h5",
                                 dim: %(hd2)d,
                                 irange: %(irange_hd2)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear  {
                                "layer_name": "h6",
                                 dim: %(hd2)d,
                                 irange: %(irange_hd2)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear {
                                "layer_name": "h7",
                                 dim: %(hd2)d,
                                 irange: %(irange_hd2)f
                            },
                            !obj:myLayer_tree_faster.myRectLinear  {
                                "layer_name": "h8",
                                 dim: %(hd2)d,
                                 irange: %(irange_hd2)f
                            }
                        ],
                    },
                    !obj:myLayer_tree_faster.myCompLayer  {
                        "layer_name": "wrapper_out",

                        "layers": [
                            !obj:myLayer_tree_faster.myLinear {
                                "layer_name": "h1",
                                 dim: %(output_dim)d,
                                 irange: %(irange_out)f
                            },
                            !obj:myLayer_tree_faster.myLinear {
                                "layer_name": "h2",
                                 dim: %(output_dim)d,
                                 irange: %(irange_out)f
                            },
                            !obj:myLayer_tree_faster.myLinear {
                                "layer_name": "h3",
                                 dim: %(output_dim)d,
                                 irange: %(irange_out)f
                            },
                            !obj:myLayer_tree_faster.myLinear {
                                "layer_name": "h4",
                                 dim: %(output_dim)d,
                                 irange: %(irange_out)f
                            },
                            !obj:myLayer_tree_faster.myLinear {
                                "layer_name": "h5",
                                 dim: %(output_dim)d,
                                 irange: %(irange_out)f
                            },
                            !obj:myLayer_tree_faster.myLinear {
                                "layer_name": "h6",
                                 dim: %(output_dim)d,
                                 irange: %(irange_out)f
                            },
                            !obj:myLayer_tree_faster.myLinear {
                                "layer_name": "h7",
                                 dim: %(output_dim)d,
                                 irange: %(irange_out)f
                            },
                            !obj:myLayer_tree_faster.myLinear {
                                "layer_name": "h8",
                                 dim: %(output_dim)d,
                                 irange: %(irange_out)f
                            },
                        ],
                    }
                ],
        },
        "algorithm": !obj:mySGD.mySGD {
            "batch_size": %(batch_size)d,
            "learning_rate": %(lamda)f,
            "cost": !obj:myLayer_tree_faster.mySumOfCosts { "costs": [
                !obj:myLayer_tree_faster.myCost {
                },
                !obj:myLayer_tree_faster.myWeightDecay {
                    "coeffs":  [ %(decay)f, %(decay)f, %(decay)f, %(decay)f , %(decay)f, %(decay)f , %(decay)f ]
                }
                ]
            },
            "termination_criterion": !obj:pylearn2.termination_criteria.And {
                "criteria":[
                    !obj:pylearn2.termination_criteria.EpochCounter {
                        "max_epochs": %(max_epochs)d,
                    }
                ]
            }
        },
        "save_path": %(save_best_path)s,
        "save_freq": 20
}

    '''

    c1 = 8
    kernel_c1 = 5
    pool_c1 = 2

    c2 = 8
    kernel_c2 = 3
    pool_c2 = 2


    lamda = 0.2
    gamma = 0.9
    decay = 0.000

    hd1 = 128
    hd2 = 36
    outdim = constants.NUM_JNTS * 3

    save_best_path = './result/faster_tree_cnn_thresh_depth3_%d_%d_%d_%d_%d.pkl'%(c1,c2, lamda*10, gamma*10, decay*1000)
    print save_best_path
    n_in = kernel_c1*kernel_c1
    n_out = c1 * kernel_c1*kernel_c1 / (pool_c1*pool_c1)
    irange_c1 = numpy.sqrt(6. / (n_in + n_out))

    n_in = c1*kernel_c2*kernel_c2
    n_out = c2 * kernel_c2*kernel_c2 / (pool_c2*pool_c2)
    irange_c2 = numpy.sqrt(6. / (n_in + n_out))

    n_in = hd1
    n_out = hd1
    irange_hd1 = numpy.sqrt(6. / (n_in + n_out))

    n_in = hd1
    n_out = hd2
    irange_hd2 = numpy.sqrt(6. / (n_in + n_out))

    n_in = hd2
    n_out = outdim
    irange_out = numpy.sqrt(6. / (n_in + n_out))

    state.hyper_parameters = {'trainfile':'train_gray_uvd_rot_1562_31.h5',
                        'N': 10*64,
                            'batch_size': 64*2,
                            'c1': c1,
                            'kernel_c1':kernel_c1,
                            'pool_c1':pool_c1,
                            'c2': c2,
                            'kernel_c2':kernel_c2,
                            'pool_c2':pool_c2,
                            'irange_c1':irange_c1,
                            'irange_c2':irange_c2,
                            'irange_hd1':irange_hd1,
                            'irange_hd2':irange_hd2,
                            'irange_out':irange_out,
                            'hd1': hd1,
                            'hd2': hd2,
                            'output_dim': outdim,
                            'lamda':lamda,
                            'decay':decay,
                            'max_epochs': 100,
                            'save_best_path': save_best_path
            }

    yaml_template = state.yaml_template
    hyper_parameters = expand(flatten(state.hyper_parameters), dict_type=ydict)
    # This will be the complete yaml string that should be executed
    final_yaml_str = yaml_template % hyper_parameters
    train_obj = pylearn2.config.yaml_parse.load(final_yaml_str)
    train_obj.main_loop()