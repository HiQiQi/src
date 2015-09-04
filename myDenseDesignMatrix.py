__author__ = 'QiYE'
import functools

import logging
import warnings

import numpy as np

from pylearn2.utils.iteration import (
    FiniteDatasetIterator,
    resolve_iterator_class
)

tables = None

logger = logging.getLogger(__name__)

from pylearn2.space import CompositeSpace, VectorSpace, IndexSpace
from pylearn2.utils.exc import reraise_as
from pylearn2.utils.rng import make_np_rng
from pylearn2.utils import contains_nan
from theano import config

from pylearn2.datasets import dense_design_matrix
from pylearn2.datasets.dense_design_matrix import DefaultViewConverter

class myDenseDesignMatrix(dense_design_matrix.DenseDesignMatrix):

    _default_seed = (17, 2, 946)

    def __init__(self, X=None, topo_view=None, y=None, latent = None,
                 view_converter=None, axes=('b', 0, 1, 'c'),
                 rng=_default_seed, preprocessor=None, fit_preprocessor=False,
                 X_labels=None, y_labels=None):

        self.latent = latent
        self.X = X
        self.y = y
        self.view_converter = view_converter
        self.X_labels = X_labels
        self.y_labels = y_labels

        self._check_labels()

        if topo_view is not None:
            assert view_converter is None
            self.set_topological_view(topo_view, axes)
        else:
            assert X is not None, ("DenseDesignMatrix needs to be provided "
                                   "with either topo_view, or X")
            if view_converter is not None:

                # Get the topo_space (usually Conv2DSpace) from the
                # view_converter
                if not hasattr(view_converter, 'topo_space'):
                    raise NotImplementedError("Not able to get a topo_space "
                                              "from this converter: %s"
                                              % view_converter)

                # self.X_topo_space stores a "default" topological space that
                # will be used only when self.iterator is called without a
                # data_specs, and with "topo=True", which is deprecated.
                self.X_topo_space = view_converter.topo_space
            else:
                self.X_topo_space = None

            # Update data specs, if not done in set_topological_view
            X_source = 'features'
            if X_labels is None:
                X_space = VectorSpace(dim=X.shape[1])
            else:
                if X.ndim == 1:
                    dim = 1
                else:
                    dim = X.shape[-1]
                X_space = IndexSpace(dim=dim, max_labels=X_labels)

            if y is None:
                space = X_space
                source = X_source
            else:
                if y.ndim == 1:
                    dim = 1
                else:
                    dim = y.shape[-1]
                if y_labels is not None:
                    y_space = IndexSpace(dim=dim, max_labels=y_labels)
                else:
                    y_space = VectorSpace(dim=dim)
                y_source = 'targets'

                Latent_space = VectorSpace(dim=latent.shape[-1])
                Latent_source = 'latents'
                space = CompositeSpace((X_space, y_space, Latent_space))
                source = (X_source, y_source, Latent_source)

            self.data_specs = (space, source)
            self.X_space = X_space

        self.compress = False
        self.design_loc = None
        self.rng = make_np_rng(rng, which_method="random_integers")
        # Defaults for iterators
        self._iter_mode = resolve_iterator_class('sequential')
        self._iter_topo = False
        self._iter_targets = False
        self._iter_data_specs = (self.X_space, 'features')

        if preprocessor:
            preprocessor.apply(self, can_fit=fit_preprocessor)
        self.preprocessor = preprocessor



    def get_data(self):
        """
        Returns all the data, as it is internally stored.
        The definition and format of these data are described in
        `self.get_data_specs()`.

        Returns
        -------
        data : numpy matrix or 2-tuple of matrices
            The data
        """
        if self.y is None:
            return self.X
        else:
            return (self.X, self.y, self.latent)



    def set_topological_view(self, V, axes=('b', 0, 1, 'c')):
        """
        Sets the dataset to represent V, where V is a batch
        of topological views of examples.

        .. todo::

            Why is this parameter named 'V'?

        Parameters
        ----------
        V : ndarray
            An array containing a design matrix representation of
            training examples.
        axes : WRITEME
        """
        assert not contains_nan(V)
        rows = V.shape[axes.index(0)]
        cols = V.shape[axes.index(1)]
        channels = V.shape[axes.index('c')]
        self.view_converter = DefaultViewConverter([rows, cols, channels],
                                                   axes=axes)
        self.X = self.view_converter.topo_view_to_design_mat(V)
        # self.X_topo_space stores a "default" topological space that
        # will be used only when self.iterator is called without a
        # data_specs, and with "topo=True", which is deprecated.
        self.X_topo_space = self.view_converter.topo_space
        assert not contains_nan(self.X)

        # Update data specs
        X_space = VectorSpace(dim=self.X.shape[1])
        X_source = 'features'
        if self.y is None:
            space = X_space
            source = X_source
        else:
            if self.y.ndim == 1:
                dim = 1
            else:
                dim = self.y.shape[-1]
            # This is to support old pickled models
            if getattr(self, 'y_labels', None) is not None:
                y_space = IndexSpace(dim=dim, max_labels=self.y_labels)
            elif getattr(self, 'max_labels', None) is not None:
                y_space = IndexSpace(dim=dim, max_labels=self.max_labels)
            else:
                y_space = VectorSpace(dim=dim)
            y_source = 'targets'

            Latent_space = VectorSpace(dim=self.latent.shape[-1])
            Latent_source = 'latents'

            space = CompositeSpace((X_space, y_space,Latent_space))
            source = (X_source, y_source,Latent_source)

        self.data_specs = (space, source)
        self.X_space = X_space
        self._iter_data_specs = (X_space, X_source)

    def get_targets(self):
        """
        .. todo::

            WRITEME
        """
        return self.y
    def get_latents(self):
        """
        .. todo::

            WRITEME
        """
        return self.latent

    def get_batch_design(self, batch_size, include_labels=False):

        try:
            idx = self.rng.randint(self.X.shape[0] - batch_size + 1)
        except ValueError:
            if batch_size > self.X.shape[0]:
                reraise_as(ValueError("Requested %d examples from a dataset "
                                      "containing only %d." %
                                      (batch_size, self.X.shape[0])))
            raise
        rx = self.X[idx:idx + batch_size, :]
        if include_labels:
            if self.y is None:
                return rx, None
            ry = self.y[idx:idx + batch_size]
            rlatent = self.latent[idx:idx + batch_size]
            return rx, ry,rlatent
        rx = np.cast[config.floatX](rx)
        return rx

    def get_batch_topo(self, batch_size, include_labels=False):
        """
        .. todo::

            WRITEME

        Parameters
        ----------
        batch_size : int
            WRITEME
        include_labels : bool
            WRITEME
        """

        if include_labels:
            batch_design, labels, latents= self.get_batch_design(batch_size, True)
        else:
            batch_design = self.get_batch_design(batch_size)

        rval = self.view_converter.design_mat_to_topo_view(batch_design)

        if include_labels:
            return rval, labels, latents

        return rval
