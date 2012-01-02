"""
@author: Ke Zhai (zhaike@cs.umd.edu)

Implements collapsed Gibbs sampling for the linear-Gaussian infinite latent feature model (IBP).
"""

import numpy, scipy;
import math, random;
from ibp.gs import GibbsSampling;

# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')

class CollapsedGibbsSampling(GibbsSampling):
    import scipy.stats;

    """
    @param data: a NxD NumPy data matrix
    @param alpha: IBP hyper parameter
    @param sigma_x: standard derivation of the noise on data, often referred as sigma_n as well
    @param sigma_a: standard derivation of the feature, often referred as sigma_f as well
    @param initializ_Z: seeded Z matrix
    """
    def _initialize(self, data, alpha=1.0, sigma_a=1.0, sigma_x=1.0, A_prior=None, initial_Z=None):
        super(CollapsedGibbsSampling, self)._initialize(self.center_data(data), alpha, sigma_a, sigma_x, initial_Z);

        if A_prior == None:
            self._A_prior = numpy.zeros((1, self._D));
        else:
            self._A_prior = A_prior;
        
        assert(self._A_prior.shape == (1, self._D));
        
        # compute matrix M
        self._M = self.compute_M();
        self._log_det_M = numpy.log(numpy.linalg.det(self._M));

        assert(numpy.abs(numpy.log(numpy.linalg.det(self._M)) - self._log_det_M) < 0.000000001)
    
    """
    sample the corpus to train the parameters
    """
    def sample(self, iteration, directory="../../output/tmp-output/"):
        import os, shutil
        if os.path.exists(directory):
            shutil.rmtree(directory);
        
        assert(self._Z.shape == (self._N, self._K));
        assert(self._X.shape == (self._N, self._D));
        
        #sample the total data
        for iter in xrange(iteration):
            # sample every object
            order = numpy.random.permutation(self._N);
            for (object_counter, object_index) in enumerate(order):
                if object_counter > 0 and object_counter % 100 == 0:
                    print("sampling in progress %2d%%" % (100 * object_counter / self._N));
                
                # compute M_i
                ziM = numpy.dot(self._Z[[object_index], :], self._M);
                ziMzi = numpy.dot(ziM, self._Z[[object_index], :].transpose());
                M_i = self._M - numpy.dot(ziM.transpose(), ziM) / (ziMzi - 1);
                log_det_M_i = self._log_det_M - numpy.log(1 - ziMzi);
                
                # sample Z_n
                singleton_features = self.sample_Zn(object_index, M_i, log_det_M_i);
                
                if self._metropolis_hastings_k_new:
                    # sample K_new using metropolis hasting
                    self.metropolis_hastings_K_new(object_index, singleton_features, M_i, log_det_M_i);
            
            self._A = self.map_estimate_A();
            
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
    def sample_Zn(self, object_index, M_i, log_det_M_i):
        assert(type(object_index) == int or type(object_index) == numpy.int32 or type(object_index) == numpy.int64);

        # calculate initial feature possess counts
        m = self._Z.sum(axis=0);
        
        # remove this data point from m vector
        new_m = (m - self._Z[object_index, :]).astype(numpy.float);
        
        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        log_prob_z1 = numpy.log(new_m / self._N);
        log_prob_z0 = numpy.log(1 - new_m / self._N);
        
        # find all singleton features possessed by current object
        singleton_features = [nk for nk in range(self._K) if self._Z[object_index, nk] != 0 and new_m[nk] == 0];
        non_singleton_features = [nk for nk in range(self._K) if nk not in singleton_features]
        order = numpy.random.permutation(self._K);

        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:
                old_Znk = self._Z[object_index, feature_index];

                # compute the log likelihood when Znk=1
                self._Z[object_index, feature_index] = 1;
                if old_Znk == 0:
                    ziMi = numpy.dot(self._Z[[object_index], :], M_i);
                    ziMizi = numpy.dot(ziMi, self._Z[[object_index], :].transpose());
                    M_tmp_1 = M_i - numpy.dot(ziMi.transpose(), ziMi) / (ziMizi + 1);
                    log_det_M_tmp_1 = log_det_M_i - numpy.log(ziMizi + 1);
                    
                    #assert(numpy.abs(log_det_M_tmp_1 - numpy.log(numpy.linalg.det(M_tmp_1))) < 0.000000001);
                else:
                    M_tmp_1 = self._M;
                    log_det_M_tmp_1 = self._log_det_M;
                    
                prob_z1 = self.log_likelihood_X(M_tmp_1, log_det_M_tmp_1);
                # add in prior
                prob_z1 += log_prob_z1[feature_index];
                
                # compute the log likelihood when Znk=0
                self._Z[object_index, feature_index] = 0;
                if old_Znk == 1:
                    ziMi = numpy.dot(self._Z[[object_index], :], M_i);
                    ziMizi = numpy.dot(ziMi, self._Z[[object_index], :].transpose());
                    M_tmp_0 = M_i - numpy.dot(ziMi.transpose(), ziMi) / (ziMizi + 1);
                    log_det_M_tmp_0 = log_det_M_i - numpy.log(ziMizi + 1);
                else:
                    M_tmp_0 = self._M;
                    log_det_M_tmp_0 = self._log_det_M;

                prob_z0 = self.log_likelihood_X(M_tmp_0, log_det_M_tmp_0);
                # add in prior
                prob_z0 += log_prob_z0[feature_index];

                #print "propose znk to 0", numpy.exp(prob_z1-prob_z0);                
                Znk_is_0 = 1 / (1 + numpy.exp(prob_z1 - prob_z0));
                #print "znk is 0 with prob", Znk_is_0
                if random.random() < Znk_is_0:
                    self._Z[object_index, feature_index] = 0;
                    self._M = M_tmp_0;
                    self._log_det_M = log_det_M_tmp_0;
                else:
                    self._Z[object_index, feature_index] = 1;
                    self._M = M_tmp_1;
                    self._log_det_M = log_det_M_tmp_1;
                    
        return singleton_features;

    """
    sample K_new using metropolis hastings algorithm
    """
    def metropolis_hastings_K_new(self, object_index, singleton_features, M_i, log_det_M_i):
        # sample K_new from the metropolis hastings proposal distribution, i.e., a poisson distribution with mean \frac{\alpha}{N}
        K_temp = scipy.stats.poisson.rvs(self._alpha / self._N);
        
        if K_temp <= 0 and len(singleton_features) <= 0:
            return False;
        
        # compute the probability of using old features
        prob_old = self.log_likelihood_X();

        # construct Z_new
        #Z_new = self._Z[:, [k for k in range(self._K) if k not in singleton_features]];
        Z_new = numpy.hstack((self._Z, numpy.zeros((self._N, K_temp))));
        Z_new[[object_index], [xrange(-K_temp, 0)]] = 1;
        Z_new[[object_index], singleton_features] = 0;

        # construct M_new
        M_i_new = numpy.vstack((numpy.hstack((M_i, numpy.zeros((self._K, K_temp)))), numpy.hstack((numpy.zeros((K_temp, self._K)), (self._sigma_a / self._sigma_x) ** 2 * numpy.eye(K_temp)))));
        log_det_M_i_new = log_det_M_i + 2 * K_temp * numpy.log(self._sigma_a / self._sigma_x);
        ziMi = numpy.dot(Z_new[[object_index], :], M_i_new);
        ziMizi = numpy.dot(ziMi, Z_new[[object_index], :].transpose());
        M_new = M_i_new - numpy.dot(ziMi.transpose(), ziMi) / (ziMizi + 1);
        log_det_M_new = log_det_M_i_new - numpy.log(ziMizi + 1);
        K_new = self._K + K_temp;
        #assert(numpy.abs(log_det_M_new - numpy.log(numpy.linalg.det(M_new))) < 0.000000001);
        
        # compute the probability of using new features
        prob_new = self.log_likelihood_X(M_new, log_det_M_new, Z_new);

        # compute the probability of generating new features
        accept_new = 1 / (1 + numpy.exp(prob_old - prob_new));
        
        # if we accept the proposal, we will replace old A and Z matrices
        if random.random() < accept_new:
            self._Z = Z_new;
            self._K = K_new;
            self.regularize_matrices();
            return True;

        return False;

    """
    remove the empty column in matrix Z and the corresponding feature in A
    """
    def regularize_matrices(self):
        assert(self._Z.shape == (self._N, self._K));
        Z_sum = numpy.sum(self._Z, axis=0);
        assert(len(Z_sum) == self._K);
        indices = numpy.nonzero(Z_sum == 0);
        
        self._Z = self._Z[:, [k for k in range(self._K) if k not in indices]];
        self._K = self._Z.shape[1];
        assert(self._Z.shape == (self._N, self._K));
        
        # compute matrix M
        self._M = self.compute_M();
        self._log_det_M = numpy.log(numpy.linalg.det(self._M));

    """
    compute the log-likelihood of the data X
    @param X: a 2-D numpy array
    @param Z: a 2-D numpy boolean array
    @param A: a 2-D numpy array, integrate A out if it is set to None
    """
    def log_likelihood_X(self, M=None, log_det_M=None, Z=None):
        if M == None:
            M = self._M;
            if log_det_M == None:
                log_det_M = numpy.log(numpy.linalg.det(M));
            else:
                log_det_M = self._log_det_M;
                
        if Z == None:
            Z = self._Z;
            
        assert(self._X.shape[0] == Z.shape[0]);
        (N, D) = self._X.shape;
        (N, K) = Z.shape;
        assert(M.shape == (K, K));
        
        # we are collapsing A out, i.e., compute the log likelihood p(X | Z)
        # be careful that M passed in should include the inverse.
        log_likelihood = numpy.eye(N) - numpy.dot(numpy.dot(Z, M), Z.transpose());
        log_likelihood = -0.5 / (self._sigma_x ** 2) * numpy.trace(numpy.dot(numpy.dot(self._X.transpose(), log_likelihood), self._X));
        log_likelihood -= D * (N - K) * numpy.log(self._sigma_x) + K * D * numpy.log(self._sigma_a);
        log_likelihood += 0.5 * D * log_det_M;
        log_likelihood -= 0.5 * N * D * numpy.log(2 * numpy.pi);
        
        return log_likelihood
    
    """
    compute the log-likelihood of the model
    """
    def log_likelihood_model(self):
        #print self.log_likelihood_X(self._X, self._Z, self._A_mean), self.log_likelihood_A(), self.log_likelihood_Z();
        return self.log_likelihood_X() + self.log_likelihood_Z();

    """
    sample noise variances, i.e., sigma_x
    """
    def sample_sigma_x(self, sigma_x_hyper_parameter):
        return self.sample_sigma(self._sigma_x_hyper_parameter, self._X - numpy.dot(self._Z, self._A));

    """
    sample feature variance, i.e., sigma_a
    """
    def sample_sigma_a(self, sigma_a_hyper_parameter):
        return self.sample_sigma(self._sigma_a_hyper_parameter, self._A);

"""
run IBP on the synthetic 'cambridge bars' dataset, used in the original paper.
"""
if __name__ == '__main__':
    import scipy.io;
    
    # load the data from the matrix
    mat_vals = scipy.io.loadmat('../../data/cambridge-bars/block_image_set.mat');
    true_weights = mat_vals['trueWeights']
    features = mat_vals['features']
    data = mat_vals['data']
    
    # set up the hyper-parameter for sampling alpha
    alpha_hyper_parameter = (1., 1.);
    # set up the hyper-parameter for sampling sigma_x
    sigma_x_hyper_parameter = (1., 1.);
    # set up the hyper-parameter for sampling sigma_a
    sigma_a_hyper_parameter = (1., 1.);
    
    features = features.astype(numpy.int);
    
    # initialize the model
    ibp = CollapsedGibbsSampling(alpha_hyper_parameter, sigma_x_hyper_parameter, sigma_a_hyper_parameter, True);

    ibp._initialize(data[1:100, :], 0.5, 0.2, 0.5);
    #ibp._initialize(data[0:1000, :], 1.0, 1.0, 1.0, None, features[0:1000, :]);
    
    #print ibp._Z, "\n", ibp._A_mean
    ibp.sample(30);
    
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
