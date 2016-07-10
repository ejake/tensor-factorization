"""
    Convex non-negative matrix factorization
    http://www.cs.berkeley.edu/~jordan/papers/ding-li-jordan-pami.pdf
"""


__authors__ = "Jose Bermeo"
__email__ = "jdbermeol@unal.edu.co"


from pylearn2.space import VectorSpace
from pylearn2.monitor import Monitor
from pylearn2.utils import sharedX
from pylearn2.utils import isfinite
from pylearn2.models import Model

from theano import tensor
from theano import function

import numpy


eps = 2.2204460492503131e-16


class CNMF(Model):

    def __init__(self, kernel, data, W,
                 lambda_vals=.0, H=None,
                 termination_criterion=None, kernel_matrix=None):
        """
            Convex non-negative matrix factorization.
            This model compute the CNMF factorization of a dataset.

            Parameters
            ----------
            kernel: Object that is going to compute the kernel between vectors.
                The object must follow the interface in kernel_two_kay_MF.kernels.
            data: Numpy matrix.
            W: Numpy matrix.
            lambda_vals: Regularization to avoid division by zero.
            H: Numpy matrix.
            termination_criterion: instance of \
                pylearn2.termination_criteria.TerminationCriterion, optional
            kernel_matrix: Numpy matrix. Represents dot product in the feature space of the data.
                If this matrix is not provided, it is going to be computed.
        """

        Model.__init__(self)

        self._kernel = kernel

        self._data = data
        if not isfinite(self._data):
            raise Exception("NaN or Inf in data")

        if kernel_matrix is not None:
            assert kernel_matrix.shape[0] == self._data.shape[0]
            self._kernel_matrix = kernel_matrix
            if not isfinite(self._kernel_matrix):
                raise Exception("NaN or Inf in kernel_matrix")
        else:
            self._compute_kernel_matrix()

        self.W = W
        if not isfinite(self.W):
            raise Exception("NaN or Inf in W")

        assert self.W.shape[1] == self._data.shape[0]

        self._data_size, self._num_features = self._data.shape
        self._num_latent_topics, _ = self.W.shape

        self.W = sharedX(self.W, name="W", borrow=True)

        if H is not None:
            if H.shape[1] != self._num_latent_topics or H.shape[0] != self._data_size:
                self.H = sharedX(
                    numpy.random.rand(
                        self._data_size,
                        self._num_latent_topics).astype(
                        self.W.dtype),
                    name="H",
                    borrow=True)
                self.init_H()
            else:
                if not isfinite(H):
                    raise Exception("NaN or Inf in H")
                else:
                    self.H = sharedX(H, name="H", borrow=True)
        else:
            self.H = sharedX(
                numpy.random.rand(self._data_size, self._num_latent_topics).astype(self.W.dtype),
                name="H", borrow=True)
            self.init_H()

        self._params = [self.W, self.H]

        self.input_space = VectorSpace(self._num_features)
        self.output_space = VectorSpace(self._num_latent_topics)

        self.lambda_vals = lambda_vals
        self._compute_update_rules()
        self._compute_helper_functions()

        Monitor.get_monitor(self)
        self.monitor._sanity_check()

        self.termination_criterion = termination_criterion

    def _compute_kernel_matrix(self):
        if not hasattr(self, "_kernel_matrix"):
            x = tensor.matrix(name="x")
            self._kernel_function = function([x], self._kernel(x, x))
            self._kernel_matrix = self._kernel_function(self._data)

    def init_H(self):
        if not hasattr(self, "_clusters"):
            a = (self.W * tensor.dot(self.W, self._kernel_matrix)).sum(axis=1) \
                - 2.0 * tensor.dot(self._kernel_matrix, self.W.T)
            b = tensor.argmin(a, axis=1)
            self._clusters = function([], b)
        H = .2 * numpy.ones((self._data_size, self._num_latent_topics)).astype(self.W.dtype)
        clusters = self._clusters()
        for i, cluster in enumerate(clusters):
            H[i, cluster] += 1.0
        self.H.set_value(H)

    def _compute_update_rules(self):
        self.kbp = 0.5 * (numpy.abs(self._kernel_matrix) + self._kernel_matrix)
        self.kbn = 0.5 * (numpy.abs(self._kernel_matrix) - self._kernel_matrix)

        a = tensor.dot(self.H, tensor.dot(self.W, self.kbn))
        b = tensor.dot(self.kbp + a, self.W.T)
        c = tensor.dot(self.H, tensor.dot(self.W, self.kbp))
        d = tensor.dot(self.kbn + c, self.W.T)
        e = self.H * tensor.sqrt(b / (d + self.lambda_vals))
        f = tensor.maximum(e, eps)

        self._update_H = function([], [], updates={self.H: f})

        a = tensor.dot(self.H, tensor.dot(self.W, self.kbn))
        b = tensor.dot(self.H.T, self.kbp + a)
        c = tensor.dot(self.H, tensor.dot(self.W, self.kbp))
        d = tensor.dot(self.H.T, self.kbn + c)
        e = self.W * tensor.sqrt(b / (d + self.lambda_vals))
        f = tensor.maximum(e, eps)

        self._update_W = function([], [], updates={self.W: f})

    def _compute_helper_functions(self):
        a = tensor.dot(self.H, self.W)
        b = -2.0 * tensor.dot(a, self._kernel_matrix.T)
        a = tensor.dot(tensor.dot(a, self._kernel_matrix), a.T)
        c = a + b + self._kernel_matrix
        d = (2.0 * self.H.shape[0]).astype(c.dtype)
        c = c.trace() / d

        self.error = function([], c)

    def set_W(self, W):
        self._num_latent_topics, data_size = W.shape
        assert data_size == self._data_size
        self.output_space = VectorSpace(self._num_latent_topics)
        self.W.set_value(W)
        if not isfinite(self.W.get_value(borrow=True)):
            raise Exception("NaN or Inf in W")

    def set_H(self, H):
        data_size, num_latent_topics = H.shape
        assert data_size == self._data_size
        assert num_latent_topics == self._num_latent_topics
        self.H.set_value(H)
        if not isfinite(self.H.get_value(borrow=True)):
            raise Exception("NaN or Inf in H")

    def train_all(self, dataset):
        """
            Train model

            Parameters
            ----------
            dataset: Pylearn dataset object.
        """

        self._update_H()
        if not isfinite(self.H.get_value(borrow=True)):
            raise Exception("NaN or Inf in H")

        self._update_W()
        if not isfinite(self.W.get_value(borrow=True)):
            raise Exception("NaN or Inf in W")

        self.monitor.report_batch(self._data_size)

    def continue_learning(self):
        if self.termination_criterion is None:
            return False
        else:
            return self.termination_criterion.continue_learning(self)

    def __call__(self, X, termination_criterion, initial_H=None):
        """
            Compute for each sample its representation.

            Parameters
            ----------
            X : Sample matrix. numpy.ndarray
            termination_criterion: pylearn TerminationCriterion object
            initial_H: Numpy matrix.

            Returns
            -------
            H: H matrix with the representation.
        """

        dataset_size = X.shape[0]

        H = None
        if initial_H is not None:
            if H.shape[0] == dataset_size and H.shape[1] == self._num_latent_topics:
                H = initial_H

        if H is None:
            if not hasattr(self, "predict_clusters"):
                h = tensor.matrix(name="h")
                x = tensor.matrix(name="x")
                kxb = self._kernel(x, self._budget)
                a = (self.W * tensor.dot(self.W, self._kernel_matrix)).sum(axis=1) \
                    - 2.0 * tensor.dot(kxb, self.W.T)
                b = tensor.argmin(a, axis=1)
                self.predict_clusters = function([x], b)

            H = .2 * numpy.ones((self._data_size, self._num_latent_topics)).astype(self.W.dtype)
            clusters = self.predict_clusters(X)
            for i, cluster in enumerate(clusters):
                H[i, cluster] += 1.0

        if not hasattr(self, "predict_representation"):
            h = tensor.matrix(name="h")
            x = tensor.matrix(name="x")
            kxb = self._kernel(x, self._budget)
            kxbp = 0.5 * (numpy.abs(kxb) + kxb)
            kxbn = 0.5 * (numpy.abs(kxb) - kxb)
            a = tensor.dot(h, tensor.dot(self.W, self.kbn))
            b = tensor.dot(kxbp + a, self.W.T)
            c = tensor.dot(h, tensor.dot(self.W, self.kbp))
            d = tensor.dot(kxbn + c, self.W.T)
            e = h * tensor.sqrt(b / (d + self.lambda_vals))
            f = tensor.maximum(e, eps)
            self.predict_representation = function([x, h], f)

        keep_training = True
        if not isfinite(H):
            raise Exception("NaN or Inf in H")

        while keep_training:
            H = self.predict_representation(X, H)
            if not isfinite(H):
                raise Exception("NaN or Inf in H")
            keep_training = termination_criterion.continue_learning(self)

        return H
