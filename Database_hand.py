import numpy
import h5py
import scipy.io as sio
import scipy
import numpy
import h5py
from pylearn2.datasets import dense_design_matrix
from myDenseDesignMatrix import myDenseDesignMatrix
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import theano.tensor as T
class Database_h5(dense_design_matrix.DenseDesignMatrix):

    def __init__(self, dataset_path, N=64*100, n=640, which_set='train', which_label = 'uvd',shuffle = False, axes=['b', 0, 1, 'c']):

        # dim is the column index to divide the pair of images
        self.args = locals()
        self.__dict__.update(locals())

        data = h5py.File(dataset_path)  # @UnusedVariable
        fea = data['gray'][...]
        id = data['uvd'][...]
        data.close()

        if N<0:
            N=fea.shape[2]

        if which_set == 'train' or which_set == 'test':
            fea = fea[0:N, :,:]
            id = id[0:N, :]
        elif which_set == 'valid':
            fea = fea[0:n,:,:]
            id = id[0:n, : ]

        fea = fea.reshape(fea.shape[0], fea.shape[1], fea.shape[2], 1)
        if which_label =='uvd':
            id = id.reshape(id.shape[0], id.shape[1]*id.shape[2])
            super(Database_h5,self).__init__(topo_view = numpy.cast['float32'](fea), y = id, axes=axes)
        else:
            u = (id[:,:,0]*96-12)/8
            v = (id[:,:,1]*96-12)/8
            heatmap = numpy.zeros((id.shape[0], id.shape[1],9, 9))
            for i in xrange(id.shape[0]):
                for j in xrange(id.shape[1]):
                    heatmap[i,j,v[i,j],u[i,j]] = 1
                # plt.imshow(heatmap[i],'gray')
                # plt.figure()
                # plt.imshow(fea[i,:,:,0],'gray')
                # plt.show()
            heatmap = heatmap.reshape(heatmap.shape[0], heatmap.shape[1]*heatmap.shape[2]*heatmap.shape[3])
            super(Database_h5,self).__init__(topo_view = numpy.cast['float32'](fea), y = heatmap, axes=axes)
        print fea.shape
        # for x in xrange(10):
        #     i = numpy.random.random_integers(0, high=fea.shape[0]/2)
        #     print label[i*2, :]
        #     plt.subplot(2,1,1)
        #     plt.imshow(fea[i*2,:,:,0], cmap=cm.Greys_r)
        #     plt.subplot(2,1,2)
        #     plt.imshow(fea[i*2+1,:,:,0], cmap=cm.Greys_r)
        #     plt.show()
        #     plt.close()
        #     c = 1


        #important: shuffle should be after the start and stop

        assert not numpy.any(numpy.isnan(self.X))

        self.error = 0

    def dimshuffle(self, b01c, default=('b', 0, 1, 'c'), axes=[]):
        return b01c.transpose(*[default.index(axis) for axis in axes])

class Database_latent(myDenseDesignMatrix):

    def __init__(self, dataset_path, N=64*100, n=640, which_set='train', which_label = 'uvd',shuffle = False, axes=['b', 0, 1, 'c']):

        # dim is the column index to divide the pair of images
        self.args = locals()
        self.__dict__.update(locals())

        data = h5py.File(dataset_path)  # @UnusedVariable
        fea = data['gray'][...]
        id = data['uvd'][...]
        latents = data['latent'][...]
        data.close()

        if N<0:
            N=fea.shape[2]

        if which_set == 'train' or which_set == 'test':
            fea = fea[0:N, :,:]
            id = id[0:N, :]
            latents = latents[0:N, :]
        elif which_set == 'valid':
            fea = fea[0:n,:,:]
            id = id[0:n, : ]
            latents = latents[0:n, :]

        fea = fea.reshape(fea.shape[0], fea.shape[1], fea.shape[2], 1)
        if which_label =='uvd':
            id = id.reshape(id.shape[0], id.shape[1]*id.shape[2])
            super(Database_latent,self).__init__(topo_view = numpy.cast['float32'](fea), y = id, latent=latents, axes=axes)
        else:
            u = (id[:,:,0]*96-12)/8
            v = (id[:,:,1]*96-12)/8
            heatmap = numpy.zeros((id.shape[0], id.shape[1],9, 9))
            for i in xrange(id.shape[0]):
                for j in xrange(id.shape[1]):
                    heatmap[i,j,v[i,j],u[i,j]] = 1
                # plt.imshow(heatmap[i],'gray')
                # plt.figure()
                # plt.imshow(fea[i,:,:,0],'gray')
                # plt.show()
            heatmap = heatmap.reshape(heatmap.shape[0], heatmap.shape[1]*heatmap.shape[2]*heatmap.shape[3])
            super(Database_latent,self).__init__(topo_view = numpy.cast['float32'](fea), y = heatmap, latent=latents,axes=axes)
        print "fea", fea.shape
        print "id", id.shape
        print "latents", latents.shape
        # for x in xrange(10):
        #     i = numpy.random.random_integers(0, high=fea.shape[0]/2)
        #     print label[i*2, :]
        #     plt.subplot(2,1,1)
        #     plt.imshow(fea[i*2,:,:,0], cmap=cm.Greys_r)
        #     plt.subplot(2,1,2)
        #     plt.imshow(fea[i*2+1,:,:,0], cmap=cm.Greys_r)
        #     plt.show()
        #     plt.close()
        #     c = 1


        #important: shuffle should be after the start and stop

        assert not numpy.any(numpy.isnan(self.X))

        self.error = 0

    def dimshuffle(self, b01c, default=('b', 0, 1, 'c'), axes=[]):
        return b01c.transpose(*[default.index(axis) for axis in axes])
# class Database_latent_multi(myMultiDenseDesignMatrix):
#
#     def __init__(self, dataset_path, N=64*100, n=640, which_set='train', which_label = 'uvd',shuffle = False, axes=['b', 0, 1, 'c']):
#
#         # dim is the column index to divide the pair of images
#         self.args = locals()
#         self.__dict__.update(locals())
#
#         data = h5py.File(dataset_path)  # @UnusedVariable
#         fea = data['gray'][...]
#         id = data['uvd'][...]
#         latents = data['latent'][...]
#         data.close()
#
#         if N<0:
#             N=fea.shape[2]
#
#         if which_set == 'train' or which_set == 'test':
#             fea = fea[0:N, :,:]
#             id = id[0:N, :]
#             latents = latents[0:N, :]
#         elif which_set == 'valid':
#             fea = fea[0:n,:,:]
#             id = id[0:n, : ]
#             latents = latents[0:n, :]
#
#         fea = fea.reshape(fea.shape[0], fea.shape[1], fea.shape[2], 1)
#         if which_label =='uvd':
#             id = id.reshape(id.shape[0], id.shape[1]*id.shape[2])
#             super(Database_latent,self).__init__(topo_view = numpy.cast['float32'](fea), y = id, latent=latents, axes=axes)
#         else:
#             u = (id[:,:,0]*96-12)/8
#             v = (id[:,:,1]*96-12)/8
#             heatmap = numpy.zeros((id.shape[0], id.shape[1],9, 9))
#             for i in xrange(id.shape[0]):
#                 for j in xrange(id.shape[1]):
#                     heatmap[i,j,v[i,j],u[i,j]] = 1
#                 # plt.imshow(heatmap[i],'gray')
#                 # plt.figure()
#                 # plt.imshow(fea[i,:,:,0],'gray')
#                 # plt.show()
#             heatmap = heatmap.reshape(heatmap.shape[0], heatmap.shape[1]*heatmap.shape[2]*heatmap.shape[3])
#             super(Database_latent,self).__init__(topo_view = numpy.cast['float32'](fea), y = heatmap, latent=latents,axes=axes)
#         print "fea", fea.shape
#         print "id", id.shape
#         print "latents", latents.shape
#         # for x in xrange(10):
#         #     i = numpy.random.random_integers(0, high=fea.shape[0]/2)
#         #     print label[i*2, :]
#         #     plt.subplot(2,1,1)
#         #     plt.imshow(fea[i*2,:,:,0], cmap=cm.Greys_r)
#         #     plt.subplot(2,1,2)
#         #     plt.imshow(fea[i*2+1,:,:,0], cmap=cm.Greys_r)
#         #     plt.show()
#         #     plt.close()
#         #     c = 1
#
#
#         #important: shuffle should be after the start and stop
#
#         assert not numpy.any(numpy.isnan(self.X))
#
#         self.error = 0
#
#     def dimshuffle(self, b01c, default=('b', 0, 1, 'c'), axes=[]):
#         return b01c.transpose(*[default.index(axis) for axis in axes])
if __name__=='__main__':
    data=Database_h5(dataset_path='test_gray_uvd_1562_31.h5',which_set='test', shuffle=True)


# class Database(dense_design_matrix.DenseDesignMatrix):
#
#     def __init__(self, dataset, dim, N = -1, n=3000, which_set='train', shuffle = False, axes=['b', 0, 1, 'c']):
#
#
#
#         # dim is the column index to divide the pair of images
#         self.args = locals()
#         self.__dict__.update(locals())
#
#         data = h5py.File(dataset, 'r')  # @UnusedVariable
#         fea = data['fea'][...]
#         id = data['id_cache'][...]
#
#         if N<0:
#             N=fea.shape[2]
#
#         if which_set == 'train' or which_set == 'test':
#             fea = fea[:,:,0:N]
#             id = id[:, 0:N]
#         elif which_set == 'valid':
#             fea = fea[:,:,0:n]
#             id = id[:, 0:n]
#
#         fea = self.dimshuffle(b01c=fea, default=[1, 0, 'b'], axes=['b', 0, 1])
#         id = id.T
#         fea = fea.reshape(fea.shape[0], fea.shape[1], fea.shape[2], 1)
#         label = numpy.zeros(shape=(fea.shape[0], dim), dtype='float32')
#
#         if which_set == 'train':
#             for i in xrange(fea.shape[0]):
#                 label[i, id[i,0]-1]=1.0
#         else:
#             label = id
#
#         if shuffle:
#             self.shuffle_rng = numpy.random.RandomState()
#             for i in xrange(fea.shape[0]/2):
#                 j = self.shuffle_rng.randint(fea.shape[0]/2)
#                 # Copy ensures that memory is not aliased.
#                 tmp = fea[i*2:i*2+1+1,:,:,:].copy()
#                 fea[i*2:i*2+1+1,:,:,:] = fea[j*2:j*2+1+1,:,:,:]
#                 fea[j*2:j*2+1+1,:,:,:] = tmp
#                 # Note: slicing with i:i+1 works for both one_hot=True/False.
#                 tmp = label[i*2:i*2+1+1,:].copy()
#                 label[i*2:i*2+1+1,:] = label[j*2:j*2+1+1,:]
#                 label[j*2:j*2+1+1,:] = tmp
#
#         # for x in xrange(10):
#         #     i = numpy.random.random_integers(0, high=fea.shape[0]/2)
#         #     print label[i*2, :]
#         #     plt.subplot(2,1,1)
#         #     plt.imshow(fea[i*2,:,:,0], cmap=cm.Greys_r)
#         #     plt.subplot(2,1,2)
#         #     plt.imshow(fea[i*2+1,:,:,0], cmap=cm.Greys_r)
#         #     plt.show()
#         #     plt.close()
#         #     c = 1
#
#         super(Database,self).__init__(topo_view = numpy.cast['float32'](fea), y = label, axes=axes)
#
#         #important: shuffle should be after the start and stop
#
#         assert not numpy.any(numpy.isnan(self.X))
#
#         self.error = 0
#
#     def dimshuffle(self, b01c, default=('b', 0, 1, 'c'), axes=[]):
#         return b01c.transpose(*[default.index(axis) for axis in axes])
