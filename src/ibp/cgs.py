"""
Author: Ke Zhai (zhaike@cs.umd.edu)

This code was modified from the code originally written by David Andrzejewski (david.andrzej@gmail.com).
Implements collapsed Gibbs sampling for the linear-Gaussian infinite latent feature model (IBP).
"""

import numpy, scipy;
import math, random;
from gs import GibbsSampling;
import util.log_math;

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
    def _initialize(self, data, alpha=1.0, sigma_x=1.0, sigma_a=1.0, A_prior=None, initial_Z=None):
        super(CollapsedGibbsSampling, self)._initialize(self.center_data(data), alpha, sigma_x, sigma_a, initial_Z);

        if A_prior==None:
            self._A_prior = numpy.zeros((1, self._D));
        else:
            self._A_prior = A_prior; 
        
        assert(self._A_prior.shape==(1, self._D));
        
        # compute matrix M
        self._M, self._det_M = self.compute_M();
         
        # initialize A from maximum a posterior estimation
        #self._A_mean = self.initialize_A_mean();
        # calculate initial feature possess counts
        #self._m = self._Z.sum(axis=0);

    """
    compute the M matrix
    """
    def compute_M(self):
        M = numpy.linalg.inv(numpy.dot(self._Z.transpose(), self._Z) + (self._sigma_x/self._sigma_a)**2*numpy.eye(self._K));
        return M, numpy.linalg.det(M);
    
    """
    initialize latent features, i.e., matrix A, randomly sample from N(0,1)
    """
    def initialize_A_mean(self):
        (mean, std_dev) = self.sufficient_statistics_A();
        assert(mean.shape==(self._K, self._D));
        
        return mean
    
#        self._A_mean = numpy.zeros((self._K, self._D));
#        # sample every feature
#        order = numpy.random.permutation(self._K);
#        for (feature_counter, feature_index) in enumerate(order):
#            # sample A_k
#            self.sample_A(feature_index);
   
    """
    sample the corpus to train the parameters
    """
    def sample(self, iteration):
        assert(self._Z.shape==(self._N, self._K));
        #assert(self._A_mean.shape==(self._K, self._D));
        assert(self._X.shape==(self._N, self._D));
        
        #sample the total data
        for iter in xrange(iteration):
            # sample every object
            order = numpy.random.permutation(self._N);
            for (object_counter, object_index) in enumerate(order):
                # compute M_i
                ziM = numpy.dot(self._Z[object_index, :], self._M);
                ziMzi = numpy.dot(ziM, self._Z[object_index, :].transpose());
                M_i = self._M - numpy.dot(ziM.transpose(), ziM) / (ziMzi-1);
                det_M_i = self._det_M / (1-ziMzi);
                
                # sample Z_n
                singleton_features = self.sample_Zn(object_index, M_i, det_M_i);
                
                if self._metropolis_hastings_k_new:
                    # sample K_new using metropolis hasting
                    self.metropolis_hastings_K_new(object_index, singleton_features, M_i, det_M_i);
                    
                # calculate initial feature possess counts
                self._m = self._Z.sum(axis=0);
            
            self.sample_A();

            # regularize matrices
            self.regularize_matrices();
            
            if self._alpha_hyper_parameter!=None:
                self._alpha = self.sample_alpha(self._alpha_hyper_parameter);
            
            if self._sigma_x_hyper_parameter!=None:
                self._sigma_x = self.sample_sigma_x(self._sigma_x_hyper_parameter);
            
            if self._sigma_a_hyper_parameter!=None:
                self._sigma_a = self.sample_sigma_a(self._sigma_a_hyper_parameter);
                
            print("iteration: %i\tK: %i\tlikelihood: %f" % (iter, self._K, self.log_likelihood_model()));
            print("alpha: %f\tsigma_a: %f\tsigma_x: %f" % (self._alpha, self._sigma_a, self._sigma_x));
          
    """
    
    @param object_index: an int data type, indicates the object index (row index) of Z we want to sample
    """
    def sample_Zn(self, object_index, M_i, det_M_i):
        assert(type(object_index)==int or type(object_index)==numpy.int32 or type(object_index)==numpy.int64);

        # calculate initial feature possess counts
        m = self._Z.sum(axis=0);
        
        # remove this data point from m vector
        new_m = (m - self._Z[object_index, :]).astype(numpy.float);
        
        # compute the log probability of p(Znk=0 | Z_nk) and p(Znk=1 | Z_nk)
        log_prob_z1 = numpy.log(new_m/self._N);
        log_prob_z0 = numpy.log(1 - new_m/self._N);
        
        # find all singleton features possessed by current object
        singleton_features = [nk for nk in range(self._K) if self._Z[object_index, nk] != 0 and new_m[nk] == 0];
        non_singleton_features = [nk for nk in range(self._K) if nk not in singleton_features]
        order = numpy.random.permutation(self._K);

        for (feature_counter, feature_index) in enumerate(order):
            if feature_index in non_singleton_features:
                old_Znk = self._Z[object_index, feature_index];

                # compute the log likelihood when Znk=1
                self._Z[object_index, feature_index]=1;
                if old_Znk==0:
                    ziMi = numpy.dot(self._Z[object_index, :], M_i);
                    ziMizi = numpy.dot(ziMi, self._Z[object_index, :].transpose());
                    M_tmp_1 = M_i - numpy.dot(ziMi.transpose(), ziMi)/(ziMizi + 1);
                    det_M_tmp_1 = det_M_i / (ziMizi+1);
                else:
                    M_tmp_1 = self._M;
                    det_M_tmp_1 = self._det_M;
                    
                prob_z1 = numpy.eye(self._N) - numpy.dot(numpy.dot(self._Z, M_tmp_1), self._Z.transpose());
                prob_z1 = -0.5/(self._sigma_x**2)*numpy.trace(numpy.dot(numpy.dot(self._X.transpose(), prob_z1), self._X))
                # take note that M matrix includes the inverse
                prob_z1 += 0.5*self._D*numpy.log(det_M_tmp_1);
                # add in prior
                prob_z1 += log_prob_z1[feature_index];          
                prob_z1 = numpy.exp(prob_z1);
                
                # compute the log likelihood when Znk=0
                self._Z[object_index, feature_index]=0;
                if old_Znk==1:
                    ziMi = numpy.dot(self._Z[object_index, :], M_i);
                    ziMizi = numpy.dot(ziMi, self._Z[object_index, :].transpose());
                    M_tmp_0 = M_i - numpy.dot(ziMi.transpose(), ziMi)/(ziMizi + 1);
                    det_M_tmp_0 = det_M_i / (ziMizi+1);
                else:
                    M_tmp_0 = self._M;
                    det_M_tmp_0 = self._det_M;
                    
                prob_z0 = numpy.eye(self._N) - numpy.dot(numpy.dot(self._Z, M_tmp_0), self._Z.transpose());
                prob_z0 = -0.5/(self._sigma_x**2)*numpy.trace(numpy.dot(numpy.dot(self._X.transpose(), prob_z0), self._X))
                # take note that M matrix includes the inverse
                prob_z0 += 0.5*self._D*numpy.log(det_M_tmp_0);
                # add in prior
                prob_z0 += log_prob_z0[feature_index];
                prob_z0 = numpy.exp(prob_z0);
                
                print prob_z0, prob_z1
                
                Znk_is_0 = prob_z0/(prob_z0+prob_z1);
                if random.random()<Znk_is_0:
                    self._Z[object_index, feature_index] = 0;
                    self._M = M_tmp_0;
                    self._det_M = det_M_tmp_0;
                else:
                    self._Z[object_index, feature_index] = 1;
                    self._M = M_tmp_1;
                    self._det_M = det_M_tmp_1;
                    
        return singleton_features;

    """
    sample K_new using metropolis hastings algorithm
    todo: finish this...
    """
    def metropolis_hastings_K_new(self, object_index, singleton_features, M_i, det_M_i):
        if type(object_index)!=list:
            object_index = [object_index];
    
        # sample K_new from the metropolis hastings proposal distribution, i.e., a poisson distribution with mean \frac{\alpha}{N}
        K_temp = scipy.stats.poisson.rvs(self._alpha / self._N);
        
        if K_temp <= 0 and len(singleton_features) <= 0:
            return False;
        
        # construct Z_old
        Z_old = self._Z;
        K_old = self._K;
        M_old = self._M;
        det_M_old = self._det_M;
        
        assert(Z_old.shape==(len(object_index), K_old));
        
        # compute the probability of using old features
        prob_old = numpy.eye(self._N)-numpy.dot(numpy.dot(Z_old, M_old), Z_old.transpose());
        prob_old = -0.5/(self._sigma_x**2) * numpy.trace(numpy.dot(numpy.dot(self._X.transpose(), prob_old), self._X));
        prob_old -= self._D*(self._N-K_old)*numpy.log(self._sigma_x) + K_old*self._D*numpy.log(self._sigma_a) + 0.5*self._D*det_M_old;        
        
        # construct Z_new
        Z_new = numpy.hstack(self._Z, numpy.zeros(self._D, K_temp));
        Z_new[object_index, [xrange(-K_temp, 0)]] = 1;
        Z_new[object_index, singleton_features] = 0;
        K_new = self._K + K_temp;
        assert(Z_new.shape==(len(object_index), K_new));

        # construct M_new
        M_i_new = numpy.hstack(numpy.hstack(M_i, numpy.zeros(self._K, K_temp)), numpy.hstack(numpy.zeros(K_temp, self._K), (self._sigma_a/self._sigma_x)**2 * numpy.eye(K_temp)));
        det_M_i_new = det_M_i / ((self._sigma_a/self._sigma_x)**(2 * K_temp));
        ziMi = numpy.dot(Z_new[object_index, :], M_i_new);
        ziMizi = numpy.dot(ziMi, Z_new[object_index, :].transpose());
        M_new = M_i_new - numpy.dot(ziMi.transpose(), ziMi)/(ziMizi + 1);
        det_M_new = det_M_i_new / (ziMizi + 1);
        
        # compute the probability of using new features
        prob_new = numpy.eye(self._N)-numpy.dot(numpy.dot(Z_new, M_new), Z_new.transpose());
        prob_new = -0.5/(self._sigma_x**2) * numpy.trace(numpy.dot(numpy.dot(self._X.transpose(), prob_new), self._X));
        prob_new -= self._D*(self._N-K_new)*numpy.log(self._sigma_x) + K_new*self._D*numpy.log(self._sigma_a) + 0.5*self._D*det_M_new;       

        # compute the probability of generating new features
        prob_new = prob_new / (prob_old + prob_new);
        
        # if we accept the proposal, we will replace old A and Z matrices
        if random.random()<prob_new:
            # construct A_new and Z_new
            self._Z = Z_new;
            self._K = K_new;
            self._M = M_new;
            self._det_M = det_M_new;
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
            assert(std_dev.shape==(self._K, self._K));
            assert(mean.shape==(self._K, len([observation_index])));
            self._A_mean[:, [observation_index]] = numpy.dot(std_dev, numpy.random.normal(0, 1, (self._K, len([observation_index])))) + mean;
        
        return
    
    """
    compute the mean and co-variance, i.e., sufficient statistics, of A
    @param observation_index: a list data type, recorded down the observation indices (column numbers) of A we want to compute
    """
    def sufficient_statistics_A(self, observation_index=None):
        if observation_index==None:
            X = self._X;
            observation_index = range(self._D);
        else:
            X = self._X[:, observation_index]
        
        assert(type(observation_index)==list);
        
        D = X.shape[1];
        #mean_a = numpy.zeros((self._K, D));
        #for k in range(self._K):
        #    mean_a[k, :] = self._mean_a[0, observation_index];
        A_prior = numpy.tile(self._A_prior[0, observation_index], (self._K, 1));

        assert(X.shape==(self._N, D));
        assert(self._Z.shape==(self._N, self._K));
        assert(A_prior.shape==(self._K, D))
        
        # compute M = (Z' * Z - (sigma_x^2) / (sigma_a^2) * I)^-1
        M = numpy.linalg.inv(numpy.dot(self._Z.transpose(), self._Z) + (self._sigma_x / self._sigma_a)**2 * numpy.eye(self._K));
        # compute the mean of the matrix A
        mean_A = numpy.dot(M, numpy.dot(self._Z.transpose(), X)-(self._sigma_x / self._sigma_a)**2 * A_prior);
        # compute the co-variance of the matrix A
        std_dev_A = numpy.linalg.cholesky(self._sigma_x**2 * M).transpose();
        
        return (mean_A, std_dev_A)
    
    """
    remove the empty column in matrix Z and the corresponding feature in A
    """
    def regularize_matrices(self):
        assert(self._Z.shape==(self._N, self._K));
        Z_sum = numpy.sum(self._Z, axis=0);
        assert(len(Z_sum)==self._K);
        indices = numpy.nonzero(Z_sum==0);
        #assert(numpy.min(indices)>=0 and numpy.max(indices)<self._K);
        
        #print self._K, indices, [k for k in range(self._K) if k not in indices]
        self._Z = self._Z[:, [k for k in range(self._K) if k not in indices]];
        self._A_mean = self._A_mean[[k for k in range(self._K) if k not in indices], :];
        
        self._K = self._Z.shape[1];
        assert(self._Z.shape==(self._N, self._K));
        assert(self._A_mean.shape==(self._K, self._D));

    """
    compute the log-likelihood of the data X
    @param X: a 2-D numpy array
    @param Z: a 2-D numpy boolean array
    @param A: a 2-D numpy array, integrate A out if it is set to None
    """
    def log_likelihood_X(self, X=None, Z=None):
        if X==None or Z==None:
            X = self._X;
            Z = self._Z;
            
        assert(X.shape[0] == Z.shape[0]);
        (N, D) = X.shape;
        (N, K) = Z.shape;

        log_likelihood = None;

        if A==None:
            # we are collapsing A out, i.e., compute the log likelihood p(X | Z)

            # compute Z^\top Z + \frac{\sigma_x^2}{\sigma_a^2} I
            M_inv = numpy.dot(Z.transpose(), Z) + numpy.eye(K)*numpy.power(self._sigma_x/self._sigma_a, 2);
            assert(M_inv.shape==(K, K));
            # compute I - Z (Z^\top Z + \frac{\sigma_x^2}{\sigma_a^2} I)^{-1} Z^\top
            log_likelihood = numpy.eye(N) - numpy.dot(numpy.dot(Z, numpy.linalg.inv(M_inv)), Z.transpose());
            assert(log_likelihood.shape==(N, N));
            # compute -\frac{1}{2 \sigma_x^2} trace(X^\top (I - Z (Z^\top Z + \frac{\sigma_x^2}{\sigma_a^2} I)^{-1} Z^\top) X)
            log_likelihood = - 0.5 * numpy.trace(numpy.dot(numpy.dot(X.transpose(), log_likelihood), X)) / numpy.power(self._sigma_x, 2);
            
            log_likelihood -= N * D * 0.5 * numpy.log(2*numpy.pi);
            log_likelihood -= (N-K) * D * numpy.log(self._sigma_x);
            log_likelihood -= K * D * numpy.log(self._sigma_a);
            log_likelihood -= 0.5 * D * numpy.log(numpy.linalg.det(M_inv));
        else:
            # we are not collapsing A out, i.e., compute the log likelihood p(X | Z, A)
            assert(A.shape==(K, D));
            
            log_likelihood = X-numpy.dot(Z, A);
            log_likelihood = - 0.5 * numpy.trace(numpy.dot(log_likelihood.transpose(), log_likelihood)) / numpy.power(self._sigma_x, 2);
            log_likelihood -= N*D*0.5*numpy.log(2 * numpy.pi * numpy.power(self._sigma_x, 2));
                       
        return log_likelihood
    
    """
    compute the log-likelihood of A
    """
    def log_likelihood_A(self):
        log_likelihood = -0.5 * self._K * self._D * numpy.log(2 * numpy.pi * self._sigma_a * self._sigma_a);
        #for k in range(self._K):
        #    A_prior[k, :] = self._mean_a[0, :];
        A_prior = numpy.tile(self._A_prior, (self._K, 1))
        log_likelihood -= numpy.trace(numpy.dot((self._A_mean-A_prior).transpose(), (self._A_mean-A_prior))) * 0.5 / (self._sigma_a**2);
        
        return log_likelihood;
    
    """
    compute the log-likelihood of the model
    """
    def log_likelihood_model(self):
        #print self.log_likelihood_X(self._X, self._Z, self._A_mean), self.log_likelihood_A(), self.log_likelihood_Z();
        return self.log_likelihood_X() + self.log_likelihood_A() + self.log_likelihood_Z();

    """
    sample noise variances, i.e., sigma_x
    """
    def sample_sigma_x(self, sigma_x_hyper_parameter):
        return self.sample_sigma(self._sigma_x_hyper_parameter, self._X - numpy.dot(self._Z, self._A_mean));
    
    """
    sample feature variance, i.e., sigma_a
    """
    def sample_sigma_a(self, sigma_a_hyper_parameter):
        return self.sample_sigma(self._sigma_a_hyper_parameter, self._A_mean - numpy.tile(self._A_prior, (self._K, 1)));
    
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
    
    # set up the hyper-parameter for sampling alpha
    alpha_hyper_parameter = (1., 1.);
    # set up the hyper-parameter for sampling sigma_x
    sigma_x_hyper_parameter = (1., 1.);
    # set up the hyper-parameter for sampling sigma_a
    sigma_a_hyper_parameter = (1., 1.);
    
    features = features.astype(numpy.int);
    
    # initialize the model
    #ibp = UncollapsedGibbsSampling(10);
    ibp = CollapsedGibbsSampling(alpha_hyper_parameter, sigma_x_hyper_parameter, sigma_a_hyper_parameter, True);
    #ibp = UncollapsedGibbsSampling(alpha_hyper_parameter);

    ibp._initialize(data, 1.0, 0.2, 1.0);
    #ibp._initialize(data[0:1000, :], 1.0, 1.0, 1.0, None, features[0:1000, :]);
    
    #print ibp._Z, "\n", ibp._A_mean
    ibp.sample(10);
    
    print ibp._Z.sum(axis=0)

    import matplotlib.pyplot
    
    # intensity plots of
    # -ground truth factor-feature weights (top)
    # -learned factor-feature weights (bottom)
    K = max(len(true_weights), len(ibp._A_mean))
    (fig, subaxes) = matplotlib.pyplot.subplots(2, K);
    for sa in subaxes.flatten():
        sa.set_visible(False)
    fig.suptitle('Ground truth (top) vs learned factors (bottom)')
    for (idx, trueFactor) in enumerate(true_weights):
        ax = subaxes[0, idx]
        ax.set_visible(True)
        util.scaled_image.scaled_image(trueFactor.reshape(6,6),
                    pixel_width=3, axes=ax)
    for (idx, learnedFactor) in enumerate(ibp._A_mean):
        ax = subaxes[1, idx]
        util.scaled_image.scaled_image(learnedFactor.reshape(6,6),
                    pixel_width=3, axes=ax)
        ax.set_visible(True)
    matplotlib.pyplot.show()