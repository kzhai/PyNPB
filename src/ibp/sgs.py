"""
@author: Ke Zhai (zhaike@cs.umd.edu)

Implements uncollapsed Gibbs sampling for the linear-Gaussian infinite latent feature model (IBP).
"""

import numpy, scipy;
import math, random;
from ibp.gs import GibbsSampling;
import scipy.stats;

# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')

class SemicollapsedGibbsSampling(GibbsSampling):
    """
    @param data: a NxD NumPy data matrix
    @param alpha: IBP hyper parameter
    @param sigma_x: standard derivation of the noise on data, often referred as sigma_n as well
    @param sigma_a: standard derivation of the feature, often referred as sigma_f as well
    @param initializ_Z: seeded Z matrix
    """
    def _initialize(self, data, alpha=1.0, sigma_f=1.0, sigma_x=1.0, initial_Z=None, A_prior=None, initial_A=None):
        # Data matrix
        super(SemicollapsedGibbsSampling, self)._initialize(self.center_data(data), alpha, sigma_f, sigma_x, A_prior, initial_Z);

        if initial_A != None:
            # this will replace the A matrix generated in the super class. 
            self._A = initial_A;
        assert(self._A.shape == (self._K, self._D));
    
    """
    sample the corpus to train the parameters
    """
    def sample(self, iteration, directory="../../output/tmp-output/"):
        import os, shutil
        if os.path.exists(directory):
            shutil.rmtree(directory);
        
        assert(self._Z.shape == (self._N, self._K));
        assert(self._A.shape == (self._K, self._D));
        assert(self._X.shape == (self._N, self._D));
        
        #sample the total data
        for iter in xrange(iteration):
            # sample every object
            order = numpy.random.permutation(self._N);
            for (object_counter, object_index) in enumerate(order):
                # sample Z_n
                singleton_features = self.sample_Zn(object_index);
                
                if self._metropolis_hastings_k_new:
                    # sample K_new using metropolis hasting
                    self.metropolis_hastings_K_new(object_index, singleton_features);

            self.sample_A();
            
            if self._alpha_hyper_parameter != None:
                self._alpha = self.sample_alpha();
            
            if self._sigma_x_hyper_parameter != None:
                self._sigma_x = self.sample_sigma_x(self._sigma_x_hyper_parameter);
            
            if self._sigma_a_hyper_parameter != None:
                self._sigma_a = self.sample_sigma_a(self._sigma_a_hyper_parameter);
                
            print("iteration: %i\tK: %i\tlikelihood: %f" % (iter, self._K, self.log_likelihood_model()));
            print("alpha: %f\tsigma_a: %f\tsigma_x: %f" % (self._alpha, self._sigma_a, self._sigma_x));
            
            if (iter + 1) % self._snapshot_interval == 0:
                self.export_snapshot(directory, iter + 1);
                
    """
    @param object_index: an int data type, indicates the object index (row index) of Z we want to sample
    """
    def sample_Zn(self, object_index):
        assert(type(object_index) == int or type(object_index) == numpy.int32 or type(object_index) == numpy.int64);
        
        # calculate initial feature possess counts
        m = self._Z.sum(axis=0);
        
        # remove this data point from m vector
        new_m = (m - self._Z[object_index, :]).astype(numpy.float);
        
        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        log_prob_z1 = numpy.log(new_m / self._N);
        log_prob_z0 = numpy.log(1.0 - new_m / self._N);
        
        # find all singleton features possessed by current object
        singleton_features = [nk for nk in range(self._K) if self._Z[object_index, nk] != 0 and new_m[nk] == 0];
        non_singleton_features = [nk for nk in range(self._K) if nk not in singleton_features]
        
        order = numpy.random.permutation(self._K);
        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:
                #old_Znk = self._Z[object_index, feature_index];

                # compute the log likelihood when Znk=0
                self._Z[object_index, feature_index] = 0;
                prob_z0 = self.log_likelihood_X(self._X[[object_index], :], self._Z[[object_index], :]);
                prob_z0 += log_prob_z0[feature_index];
                prob_z0 = numpy.exp(prob_z0);
                
                # compute the log likelihood when Znk=1
                self._Z[object_index, feature_index] = 1;
                prob_z1 = self.log_likelihood_X(self._X[[object_index], :], self._Z[[object_index], :]);
                prob_z1 += log_prob_z1[feature_index]
                prob_z1 = numpy.exp(prob_z1);
                
                Znk_is_0 = prob_z0 / (prob_z0 + prob_z1);
                if random.random() < Znk_is_0:
                    self._Z[object_index, feature_index] = 0;
                else:
                    self._Z[object_index, feature_index] = 1;
                    
        return singleton_features;

    """
    sample K_new using metropolis hastings algorithm
    """
    def metropolis_hastings_K_new(self, object_index, singleton_features):
        if type(object_index) != list:
            object_index = [object_index];
    
        # sample K_new from the metropolis hastings proposal distribution, i.e., a poisson distribution with mean \frac{\alpha}{N}
        K_new = scipy.stats.poisson.rvs(self._alpha / self._N);
        K_old = len(singleton_features);
        
        if K_new <= 0 and K_old <= 0:
            return False;
        
        Z_i_tmp = self._Z[object_index, :];
        Z_i_tmp[:, singleton_features] = 0;
        X_residue = self._X[object_index, :] - numpy.dot(Z_i_tmp, self._A);

        log_new_old = 0;
        for d in xrange(self._D):
            log_new_old -= 0.5 * numpy.log((self._sigma_x ** 2 + K_new * self._sigma_a ** 2) / (self._sigma_x ** 2 + K_old * self._sigma_a ** 2));
            log_new_old -= 0.5 * X_residue[0, d] ** 2 * (1 / (self._sigma_x ** 2 + K_new * self._sigma_a ** 2) - 1 / (self._sigma_x ** 2 + K_old * self._sigma_a ** 2));
            
        accept_new = 1.0 / (1.0 + 1.0 / numpy.exp(log_new_old));

        '''
        # compute the log likelihood if we use old features
        Z_i_old = numpy.ones((1, K_old));
        M_old = self.compute_M(Z_i_old);
        assert(M_old.shape==(K_old, K_old));
        log_likelihood_old = 1-numpy.dot(numpy.dot(Z_i_old, M_old), Z_i_old.transpose());
        log_likelihood_old = -numpy.trace(numpy.dot(numpy.dot(X_residue.transpose(), log_likelihood_old), X_residue));
        log_likelihood_old /= (2 * self._sigma_x**2);
        log_likelihood_old += self._D / 2 * numpy.linalg.det(M_old);
        log_likelihood_old -= (1-K_old)*self._D * numpy.log(self._sigma_x) + (K_old*self._D) * numpy.log(self._sigma_a);
        
        # compute the log likelihood if we use new features
        Z_i_new = numpy.ones((1, K_new));
        M_new = self.compute_M(Z_i_new);
        assert(M_new.shape==(K_new, K_new));
        log_likelihood_new = 1-numpy.dot(numpy.dot(Z_i_new, M_new), Z_i_new.transpose());
        log_likelihood_new = -numpy.trace(numpy.dot(numpy.dot(X_residue.transpose(), log_likelihood_new), X_residue));
        log_likelihood_new /= (2 * self._sigma_x**2);
        log_likelihood_new += self._D / 2 * numpy.linalg.det(M_new);
        log_likelihood_new -= (1-K_new)*self._D * numpy.log(self._sigma_x) + (K_new*self._D) * numpy.log(self._sigma_a);
        
        # compute the probability of accepting new features                
        accept_new = 1.0/(1.0 + numpy.exp(log_likelihood_old-log_likelihood_new));
        '''

        self._A = self._A[[k for k in xrange(self._K) if k not in singleton_features], :];
        self._Z = self._Z[:, [k for k in xrange(self._K) if k not in singleton_features]];
        self._K -= K_old
        
        # if we accept the proposal, we will replace old A and Z matrices
        if random.random() > accept_new:
            K_new = K_old;
        
        if K_new > 0:
            # construct A_new and Z_new
            Z_new = numpy.zeros((self._N, K_new));
            Z_i_new = numpy.ones((1, K_new));
            Z_new[object_index, :] = Z_i_new;
            M_new = self.compute_M(Z_i_new);
            A_new = numpy.dot(M_new, numpy.dot(Z_new.transpose(), self._X - numpy.dot(self._Z, self._A)));
            self._A = numpy.vstack((self._A, A_new));
            self._Z = numpy.hstack((self._Z, Z_new));
            #self._Z[object_index, :] = Z_i_new;
            self._K += K_new
            return True;

        return False;

    """
    """
    def sample_A(self):
        # sample every feature
        order = numpy.random.permutation(self._D);
        for (observation_counter, observation_index) in enumerate(order):
            # sample A_d
            (mean, std_dev) = self.sufficient_statistics_A([observation_index]);
            assert(std_dev.shape == (self._K, self._K));
            assert(mean.shape == (self._K, len([observation_index])));
            self._A[:, [observation_index]] = numpy.dot(std_dev, numpy.random.normal(0, 1, (self._K, len([observation_index])))) + mean;
        
        return
    
    """
    compute the mean and co-variance, i.e., sufficient statistics, of A
    @param observation_index: a list data type, recorded down the observation indices (column numbers) of A we want to compute
    """
    def sufficient_statistics_A(self, observation_index=None):
        if observation_index == None:
            X = self._X;
            observation_index = range(self._D);
        else:
            X = self._X[:, observation_index]
        
        assert(type(observation_index) == list);
        
        D = X.shape[1];
        #mean_a = numpy.zeros((self._K, D));
        #for k in range(self._K):
        #    mean_a[k, :] = self._mean_a[0, observation_index];
        A_prior = numpy.tile(self._A_prior[0, observation_index], (self._K, 1));

        assert(X.shape == (self._N, D));
        assert(self._Z.shape == (self._N, self._K));
        assert(A_prior.shape == (self._K, D))
        
        # compute M = (Z' * Z - (sigma_x^2) / (sigma_a^2) * I)^-1
        M = self.compute_M();
        # compute the mean of the matrix A
        mean_A = numpy.dot(M, numpy.dot(self._Z.transpose(), X) + (self._sigma_x / self._sigma_a) ** 2 * A_prior);
        # compute the co-variance of the matrix A
        std_dev_A = numpy.linalg.cholesky(self._sigma_x ** 2 * M).transpose();
        
        return (mean_A, std_dev_A)
    
    """
    remove the empty column in matrix Z and the corresponding feature in A
    """
    def regularize_matrices(self):
        assert(self._Z.shape == (self._N, self._K));
        Z_sum = numpy.sum(self._Z, axis=0);
        assert(len(Z_sum) == self._K);
        indices = numpy.nonzero(Z_sum == 0);
        #assert(numpy.min(indices)>=0 and numpy.max(indices)<self._K);
        
        #print self._K, indices, [k for k in range(self._K) if k not in indices]
        self._Z = self._Z[:, [k for k in range(self._K) if k not in indices]];
        self._A = self._A[[k for k in range(self._K) if k not in indices], :];
        
        self._K = self._Z.shape[1];
        assert(self._Z.shape == (self._N, self._K));
        assert(self._A.shape == (self._K, self._D));

    """
    compute the log-likelihood of the data X
    @param X: a 2-D numpy array
    @param Z: a 2-D numpy boolean array
    @param A: a 2-D numpy array, integrate A out if it is set to None
    """
    def log_likelihood_X(self, X=None, Z=None, A=None):
        if A == None:
            A = self._A;
        if Z == None:
            Z = self._Z;
        if X == None:
            X = self._X;
            
        assert(X.shape[0] == Z.shape[0]);
        (N, D) = X.shape;
        (N, K) = Z.shape;
        assert(A.shape == (K, D));
        
        log_likelihood = X - numpy.dot(Z, A);
        
        (row, column) = log_likelihood.shape;
        if row > column:
            log_likelihood = numpy.trace(numpy.dot(log_likelihood.transpose(), log_likelihood));
        else:
            log_likelihood = numpy.trace(numpy.dot(log_likelihood, log_likelihood.transpose()));
        
        log_likelihood = -0.5 * log_likelihood / numpy.power(self._sigma_x, 2);
        log_likelihood -= N * D * 0.5 * numpy.log(2 * numpy.pi * numpy.power(self._sigma_x, 2));
                       
        return log_likelihood
    
    """
    compute the log-likelihood of A
    """
    def log_likelihood_A(self):
        log_likelihood = -0.5 * self._K * self._D * numpy.log(2 * numpy.pi * self._sigma_a * self._sigma_a);
        #for k in range(self._K):
        #    A_prior[k, :] = self._mean_a[0, :];
        A_prior = numpy.tile(self._A_prior, (self._K, 1))
        log_likelihood -= numpy.trace(numpy.dot((self._A - A_prior).transpose(), (self._A - A_prior))) * 0.5 / (self._sigma_a ** 2);
        
        return log_likelihood;
    
    """
    compute the log-likelihood of the model
    """
    def log_likelihood_model(self):
        #print self.log_likelihood_X(self._X, self._Z, self._A), self.log_likelihood_A(), self.log_likelihood_Z();
        return self.log_likelihood_X() + self.log_likelihood_A() + self.log_likelihood_Z();

    """
    sample noise variances, i.e., sigma_x
    """
    def sample_sigma_x(self, sigma_x_hyper_parameter):
        return self.sample_sigma(self._sigma_x_hyper_parameter, self._X - numpy.dot(self._Z, self._A));
    
    """
    sample feature variance, i.e., sigma_a
    """
    def sample_sigma_a(self, sigma_a_hyper_parameter):
        return self.sample_sigma(self._sigma_a_hyper_parameter, self._A - numpy.tile(self._A_prior, (self._K, 1)));
    
"""
run IBP on the synthetic 'cambridge bars' dataset, used in the original paper.
"""
if __name__ == '__main__':
    import scipy.io;
    #import util.scaled_image;
    
    # load the data from the matrix
    mat_vals = scipy.io.loadmat('../../data/cambridge-bars/block_image_set.mat');
    true_weights = mat_vals['trueWeights']
    features = mat_vals['features']
    data = mat_vals['data']
    
    print true_weights.shape, features.shape, data.shape
    
    # set up the hyper-parameter for sampling alpha
    alpha_hyper_parameter = (1., 1.);
    # set up the hyper-parameter for sampling sigma_x
    sigma_x_hyper_parameter = (1., 1.);
    # set up the hyper-parameter for sampling sigma_a
    sigma_a_hyper_parameter = (1., 1.);
    
    features = features.astype(numpy.int);
    
    # initialize the model
    ibp = SemicollapsedGibbsSampling(alpha_hyper_parameter, sigma_x_hyper_parameter, sigma_a_hyper_parameter, True);

    ibp._initialize(data[1:100, :], 0.5, 0.2, 0.5, None, None, None);

    ibp.sample(20);
    
    print ibp._Z.sum(axis=0)

    # If matplotlib is installed, plot ground truth vs learned factors
    import matplotlib.pyplot as P
    from util.scaled_image import scaledimage
    
    # Intensity plots of
    # -ground truth factor-feature weights (top)
    # -learned factor-feature weights (bottom)
    K = max(len(true_weights), len(ibp._A))
    (fig, subaxes) = P.subplots(2, K)
    for sa in subaxes.flatten():
        sa.set_visible(False)
    fig.suptitle('Ground truth (top) vs learned factors (bottom)')
    for (idx, trueFactor) in enumerate(true_weights):
        ax = subaxes[0, idx]
        ax.set_visible(True)
        scaledimage(trueFactor.reshape(6, 6),
                    pixwidth=3, ax=ax)
    for (idx, learnedFactor) in enumerate(ibp._A):
        ax = subaxes[1, idx]
        scaledimage(learnedFactor.reshape(6, 6),
                    pixwidth=3, ax=ax)
        ax.set_visible(True)
    P.show() 
