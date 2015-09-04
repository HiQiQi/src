__author__ = 'QiYE'
import theano
import theano.tensor as T

from Database_hand import Database_latent
from pylearn2.utils import serial
from pylearn2.space import *
import constants
import scipy.io as sio

if __name__ == '__main__':
    c1 = 8
    kernel_c1 = 5
    pool_c1 = 2

    c2 = 8
    kernel_c2 = 3
    pool_c2 = 2


    lamda = 0.1
    gamma = 0.0
    decay = 0.000

    hd1 = 128
    hd2 = 36
    outdim = constants.NUM_JNTS * 3

    nbatch = 2
    batch_size = 64
    model = serial.load('./result/tree_cnn_thresh_depth3_%d_%d_%d_%d_%d.pkl'%(c1,c2, lamda*10, gamma*10, decay*1000))

    data_tmp = Database_latent(dataset_path='test_gray_uvd_rot_1562_31.h5', which_set='test', shuffle=False, N=nbatch*batch_size)
    idx_below = T.arange(0,batch_size,1)

    x0=T.tensor4('x0', dtype=theano.config.floatX)
    y0=T.fmatrix('y0')
    latents = T.fmatrix('latent0')

    tmp_out0 = model.layers[0].fprop(x0)
    tmp_out11, idx_below, latents = model.layers[1].tree_fprop(tmp_out0,idx_below,latents)
    tmp_out12, idx_below, latents = model.layers[2].tree_fprop(tmp_out11,idx_below,latents)
    tmp_out13, idx_below, latents = model.layers[3].tree_fprop(tmp_out12,idx_below,latents)


    tmp_out20 = model.layers[4].myfprop(tmp_out13)
    tmp_out21 = model.layers[5].myfprop(tmp_out20)
    tmp_out_y = model.layers[6].myfprop(tmp_out21)


    cost_list = model.layers[6].cost(y0, tmp_out_y,idx_below)
    cost_vector = T.as_tensor_variable(cost_list)
    fn = theano.function([x0,y0,latents], [tmp_out_y,cost_vector,idx_below])

    for j in xrange(0, model.batch_size*nbatch, model.batch_size):
        print j
        tmp_in = data_tmp.get_topological_view(data_tmp.X[j:j+model.batch_size, :])
        tmp_y = data_tmp.y[j:j+model.batch_size, :]
        tmp_latent = data_tmp.latent[j:j+model.batch_size, :]

        tmp_out,tmp_cost_list, tmp_idx = fn(tmp_in,tmp_y,tmp_latent)
        print tmp_latent
        print tmp_idx

        if j==0:
            cost_sum_vect = tmp_cost_list
            tmp=tmp_out
        else:
            tmp=np.concatenate((tmp, tmp_out), axis=0)

            cost_sum_vect += tmp_cost_list


    cost_sum_vect /= nbatch
    print cost_sum_vect
