from pylearn2.train import Train
from pylearn2.monitor import Monitor
from pylearn2.termination_criteria import EpochCounter, And
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix

import numpy

from semantic_methods_toolkit.python.kernel_semantic_embedding_methods.model.\
    convex_non_negative_matrix_factorization import CNMF

from semantic_methods_toolkit.python.kernel_semantic_embedding_methods.model.online_kmeans \
    import OnlineKMeans

from semantic_methods_toolkit.python.kernel_semantic_embedding_methods.model.kernel_kmeans \
    import KernelKMeans

from semantic_methods_toolkit.python.kernel_semantic_embedding_methods.termination_criteria.\
    cnmf_termination_criteria import CNMFTerminationCriterion

from semantic_methods_toolkit.python.utils.kernel_factory import *


class CnmfExperiment():

    def __init__(self, X, options):
        self.reset(X, options)

    def reset(self, X, options):
        # Number of latent topics
        self.num_latent_topics = options["k"]

        # data size
        self.num_samples, _ = X.shape

        # Termination critirion
        self.termination_critirion_init(X, options)

        # Kernel
        kernel = kernel_factory(options)

        if hasattr(self, "kernel"):
            if self.kernel is not kernel:
                self.kernel = None
        else:
            self.kernel = None

        if hasattr(self, "X"):
            if self.X is not X:
                self.X = None
        else:
            self.X = None

        if self.X is None or self.kernel is None:
            self.X = X
            self.kernel = kernel
            if "lambda_vals" in options:
                self.cnmf = CNMF(
                    self.kernel, self.X, numpy.ones(
                        (self.num_latent_topics, self.num_samples)).astype(
                        X.dtype), lambda_vals=options["lambda_vals"])
            else:
                self.cnmf = CNMF(
                    self.kernel, self.X, numpy.ones(
                        (self.num_latent_topics, self.num_samples)).astype(
                        X.dtype))

        # Matrix initialization
        self.matrix_init(X, options)

    def termination_critirion_init(self, X, options):
        if "termination_criterion" in options:
            tc_options = options["termination_criterion"]
            critiria = []
            if "tol" in tc_options:
                if "tol_frequency" in tc_options:
                    critiria.append(CNMFTerminationCriterion(
                        tc_options["tol"],
                        frequency=int(tc_options["tol_frequency"])))
                else:
                    critiria.append(
                        CNMFTerminationCriterion(
                            tc_options["tol"]))
            if "iter" in tc_options:
                critiria.append(
                    EpochCounter(
                        max_epochs=int(
                            tc_options["iter"])))
            else:
                critiria.append(EpochCounter(max_epochs=10))
            self.termination_criterion = And(critiria)
        else:
            self.termination_criterion = EpochCounter(max_epochs=10)

    def matrix_init(self, X, options):
        if "initialization" in options:
            initialization = options["initialization"].strip().lower()
        else:
            initialization = "deterministic"

        if initialization == "random":
            self.W = .2 * numpy.random.rand(self.num_latent_topics,
                                            self.num_samples).astype(X.dtype)
            self.H = .2 * \
                numpy.random.rand(self.num_samples, self.num_latent_topics).astype(X.dtype)
        elif initialization == "deterministic":
            self.W = .2 * \
                numpy.ones((self.num_latent_topics, self.num_samples)).astype(X.dtype)
            self.H = .2 * \
                numpy.ones((self.num_samples, self.num_latent_topics)).astype(X.dtype)
        elif initialization == "kmeans":
            kmeans = OnlineKMeans(self.num_latent_topics)
            _, clusters = kmeans.train_all(X, epochs=10)
            self.W = .2 * \
                numpy.ones((self.num_latent_topics, self.num_samples)).astype(X.dtype)
            for i, cluster in enumerate(clusters):
                self.W[cluster, i] += 1.0
            self.cnmf.init_H()
        elif initialization == "kkmeans":
            kmeans = KernelKMeans(self.num_latent_topics, self.kernel)
            kmeans.train_all(self.X,
                             termination_criterion=EpochCounter(max_epochs=10),
                             kernel_matrix=self.cnmf._kernel_matrix)
            self.W = numpy.asarray(kmeans.get_weights())
            self.cnmf.init_H()

    def __call__(self):
        dataset = DenseDesignMatrix(X=self.X)
        self.cnmf.termination_criterion = self.termination_criterion
        self.cnmf.set_W(self.W)
        train = Train(dataset, self.cnmf)
        train.main_loop()
        self.cnmf.monitor = Monitor(self.cnmf)
        H = self.cnmf.H.get_value()
        results = {"W": self.cnmf.W.get_value(), "H": H}
        return numpy.argmax(H, axis=1), results
