"""
Author: Ke Zhai (zhaike@cs.umd.edu)

This code was modified from the code originally written by David Andrzejewski (david.andrzej@gmail.com).
Implements collapsed Gibbs sampling for the linear-Gaussian infinite latent feature model (IBP).
"""

import numpy, scipy;
import math, random;
import util.log_math;

# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')

class CollapsedGibbsSampling(object):
    import scipy.stats;
    
    """
    @param gibbs_sampling_maximum_iteration: gibbs sampling maximum iteration
    @param alpha_hyper_parameter: hyper-parameter for alpha sampling, a tuple defining the parameter for an inverse gamma distribution
    @param sigma_a_hyper_parameter: hyper-parameter for sigma_a sampling, a tuple defining the parameter for an inverse gamma distribution
    @param sigma_x_hyper_parameter: hyper-parameter for sigma_x sampling, a tuple defining the parameter for an inverse gamma distribution
    @param metropolis_hasting_k_new: a boolean variable, set to true if we use metropolis hasting to estimate K_new, otherwise use truncated gibbs sampling
    """
    def __init__(self, #real_valued_latent_feature=True,
                 alpha_hyper_parameter=None, 
                 sigma_a_hyper_parameter=None, 
                 sigma_x_hyper_parameter=None,
                 metropolis_hastings_k_new=True):
        # initialize the hyper-parameter for sampling _alpha
        # a value of None is a gentle way to say "do not sampling _alpha"
        assert(alpha_hyper_parameter==None or type(alpha_hyper_parameter)==tuple);
        self._alpha_hyper_parameter = alpha_hyper_parameter;
        # initialize the hyper-parameter for sampling _sigma_x
        # a value of None is a gentle way to say "do not sampling _sigma_x"
        assert(sigma_x_hyper_parameter==None or type(sigma_x_hyper_parameter)==tuple);
        self._sigma_x_hyper_parameter = sigma_x_hyper_parameter;
        # initialize the hyper-parameter for sampling _sigma_a
        # a value of None is a gentle way to say "do not sampling _sigma_a"
        assert(sigma_a_hyper_parameter==None or type(sigma_a_hyper_parameter)==tuple);
        self._sigma_a_hyper_parameter = sigma_a_hyper_parameter;
        
        #self._real_valued_latent_feature = real_valued_latent_feature;
        self._metropolis_hastings_k_new = metropolis_hastings_k_new;
    
    """
    @param data: a NxD NumPy data matrix
    @param alpha: IBP hyper parameter
    @param sigma_x: standard derivation of the noise on data, often referred as sigma_n as well
    @param sigma_a: standard derivation of the feature, often referred as sigma_f as well
    @param initializ_Z: seeded Z matrix
    """
    def _initialize(self, data, alpha=1.0, sigma_x=1.0, sigma_a=1.0, A_prior=None, initial_Z=None):
        self._alpha = alpha;
        self._sigma_x = sigma_x;
        self._sigma_a = sigma_a;

        # Data matrix
        self._X = self.center_data(data);
        (self._N, self._D) = self._X.shape;
        

        if A_prior==None:
            self._A_prior = numpy.zeros((1, self._D));
        else:
            self._A_prior = A_prior; 
        
        assert(self._A_prior.shape==(1, self._D));
        
        if(initial_Z == None):
            # initialize Z from IBP(alpha)
            self._Z = self.initialize_Z();
        else:
            self._Z = initial_Z;
            
        assert(self._Z.shape[0]==self._N);
        
        # make sure Z matrix is a binary matrix
        assert(self._Z.dtype==numpy.int);
        assert(self._Z.max()==1 and self._Z.min()==0);    
                
        # record down the number of features
        self._K = self._Z.shape[1];
        
        # initialize A from maximum a posterior estimation
        self._A_mean = self.initialize_A_mean();
        # calculate initial feature possess counts
        #self._m = self._Z.sum(axis=0);
    
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
    initialize latent feature appearance matrix Z according to IBP(alpha)
    """
    def initialize_Z(self):
        Z = numpy.ones((0,0));
        # initialize matrix Z recursively in IBP manner
        for i in xrange(1,self._N+1):
            # sample existing features
            # Z.sum(axis=0)/i: compute the popularity of every dish, computes the probability of sampling that dish
            sample_dish = (numpy.random.uniform(0,1,(1,Z.shape[1])) < (Z.sum(axis=0).astype(numpy.float) / i));
            # sample a value from the poisson distribution, defines the number of new features
            K_new = scipy.stats.poisson.rvs((self._alpha / i));
            # horizontally stack or append the new dishes to current object's observation vector, i.e., the vector Z_{n*}
            sample_dish = numpy.hstack((sample_dish, numpy.ones((1, K_new))));
            # append the matrix horizontally and then vertically to the Z matrix
            Z = numpy.hstack((Z, numpy.zeros((Z.shape[0], K_new))));
            Z = numpy.vstack((Z, sample_dish));
            
        assert(Z.shape[0]==self._N);
        Z = Z.astype(numpy.int);
        
        return Z
    
    """
    sample the corpus to train the parameters
    """
    def sample(self, iteration):
        assert(self._Z.shape==(self._N, self._K));
        assert(self._A_mean.shape==(self._K, self._D));
        assert(self._X.shape==(self._N, self._D));
        
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
    def sample_Zn(self, object_index):
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
                #old_Znk = self._Z[object_index, feature_index];

                # compute the log likelihood when Znk=0
                self._Z[object_index, feature_index]=0;
                prob_z0 = numpy.exp(self.log_likelihood_X(numpy.array([self._X[object_index, :]]), numpy.array([self._Z[object_index, :]]), self._A_mean) + log_prob_z0[feature_index]);
                
                # compute the log likelihood when Znk=1
                self._Z[object_index, feature_index]=1;
                prob_z1 = numpy.exp(self.log_likelihood_X(numpy.array([self._X[object_index, :]]), numpy.array([self._Z[object_index, :]]), self._A_mean) + log_prob_z1[feature_index]);
                
                #print prob_z0, prob_z1
                
                Znk_is_0 = prob_z0/(prob_z0+prob_z1);
                if random.random()<Znk_is_0:
                    self._Z[object_index, feature_index] = 0;
                else:
                    self._Z[object_index, feature_index] = 1;
                    
        return singleton_features;

    """
    sample K_new using metropolis hastings algorithm
    """
    def metropolis_hastings_K_new(self, object_index, singleton_features):
        if type(object_index)!=list:
            object_index = [object_index];
    
        # sample K_new from the metropolis hastings proposal distribution, i.e., a poisson distribution with mean \frac{\alpha}{N}
        K_temp = scipy.stats.poisson.rvs(self._alpha / self._N);
        
        if K_temp <= 0 and len(singleton_features) <= 0:
            return False;

        A_prior = numpy.tile(self._A_prior, (K_temp, 1));

        # generate new features from a normal distribution with mean 0 and variance sigma_a, a K_new-by-D matrix
        A_temp = numpy.random.normal(0, self._sigma_a, (K_temp, self._D)) + A_prior;
        A_new = numpy.vstack((self._A_mean[[k for k in xrange(self._K) if k not in singleton_features], :], A_temp));
        # generate new z matrix row
        Z_new = numpy.hstack((numpy.array([self._Z[object_index, [k for k in xrange(self._K) if k not in singleton_features]]]), numpy.ones((len(object_index), K_temp))));
        
        K_new = self._K + K_temp - len(singleton_features);
        
        # compute the probability of generating new features
        prob_new = numpy.exp(self.log_likelihood_X(self._X[object_index, :], Z_new, A_new));
        
        # construct the A_old and Z_old
        A_old = self._A_mean;
        Z_old = self._Z[object_index, :];

        K_old = self._K;
        assert(A_old.shape==(K_old, self._D));
        assert(A_new.shape==(K_new, self._D));
        assert(Z_old.shape==(len(object_index), K_old));
        assert(Z_new.shape==(len(object_index), K_new));
        
        # compute the probability of using old features
        prob_old = numpy.exp(self.log_likelihood_X(self._X[object_index, :], Z_old, A_old));
        
        # compute the probability of generating new features
        prob_new = prob_new / (prob_old + prob_new);
        
        # if we accept the proposal, we will replace old A and Z matrices
        if random.random()<prob_new:
            # construct A_new and Z_new
            self._A_mean = A_new;
            self._Z = numpy.hstack((self._Z[:, [k for k in xrange(self._K) if k not in singleton_features]], numpy.zeros((self._N, K_temp))));
            self._Z[object_index, :] = Z_new;
            self._K = K_new;
            
        return True;

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
    def log_likelihood_X(self, X=None, Z=None, A=None):
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
    compute the log-likelihood of the Z matrix.
    """
    def log_likelihood_Z(self):
        # compute {K_+} \log{\alpha} - \alpha * H_N, where H_N = \sum_{j=1}^N 1/j
        H_N = numpy.array([range(self._N)])+1.0;
        H_N = numpy.sum(1.0/H_N);
        log_likelihood = self._K * numpy.log(self._alpha) - self._alpha * H_N;
        
        # compute the \sum_{h=1}^{2^N-1} \log{K_h!}
        Z_h = numpy.sum(self._Z, axis=0).astype(numpy.int);
        Z_h = list(Z_h);
        for k_h in set(Z_h):
            log_likelihood -= numpy.log(math.factorial(Z_h.count(k_h)));
            
        # compute the \sum_{k=1}^{K_+} \frac{(N-m_k)! (m_k-1)!}{N!}
        for k in xrange(self._K):
            m_k = Z_h[k];
            temp_var = 1.0;
            if m_k-1<self._N-m_k:
                for k_prime in range(self._N-m_k+1, self._N+1):
                    if m_k!=1:
                        m_k -= 1;
                        
                    temp_var /= k_prime;
                    temp_var *= m_k;
            else:
                n_m_k = self._N - m_k;
                for k_prime in range(m_k, self._N+1):
                    temp_var /= k_prime;
                    temp_var += n_m_k;
                    if n_m_k!=1:
                        n_m_k -= 1;
            
            log_likelihood += numpy.log(temp_var);            

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
    sample standard deviation of a multivariant Gaussian distribution
    @param sigma_hyper_parameter: the hyper-parameter of the gamma distribution
    @param matrix: a r*c matrix drawn from a multivariant c-dimensional Gaussian distribution with zero mean and identity c*c covariance matrix
    """
    @staticmethod
    def sample_sigma(sigma_hyper_parameter, matrix):
        assert(sigma_a_hyper_parameter!=None);
        assert(matrix!=None);
        assert(type(sigma_hyper_parameter)==tuple);
        assert(type(matrix)==numpy.ndarray);
        
        (sigma_hyper_a, sigma_hyper_b) = sigma_hyper_parameter;
        (row, column) = matrix.shape;

        # compute the posterior_shape = sigma_hyper_a + n/2, where n = self._D * self._K
        posterior_shape = sigma_hyper_a + 0.5 * row * column;
        # compute the posterior_scale = sigma_hyper_b + sum_{k} (A_k - \mu_A)(A_k - \mu_A)^\top/2
        var = 0;
        if row>=column:
            var = numpy.trace(numpy.dot(matrix.transpose(), matrix));
        else:
            var = numpy.trace(numpy.dot(matrix, matrix.transpose()));
        
        posterior_scale = 1.0/(sigma_hyper_b + var*0.5);
        tau = scipy.stats.gamma.rvs(posterior_shape,scale=posterior_scale);
        sigma_a_new = numpy.sqrt(1.0/tau);
        
        return sigma_a_new;
    
    """
    sample alpha from conjugate posterior
    """
    def sample_alpha(self, alpha_hyper_parameter):        
        assert(alpha_hyper_parameter!=None);
        (alpha_hyper_a, alpha_hyper_b) = alpha_hyper_parameter;
        
        (N, D) = self._X.shape;
        (N, K) = self._Z.shape;
        
        posterior_shape = alpha_hyper_a + K;
        
        H_N = numpy.array([range(self._N)])+1.0;
        H_N = numpy.sum(1.0/H_N);
        posterior_scale = 1.0/(alpha_hyper_b + H_N);
     
        alpha_new = scipy.stats.gamma.rvs(posterior_shape,scale=posterior_scale);
        
        return alpha_new;
    
    """
    center the data, i.e., subtract the mean
    """
    @staticmethod
    def center_data(data):
        (N, D) = data.shape;
        data = (data - data.mean())/data.std();
        return data
    
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
    ibp = UncollapsedGibbsSampling(alpha_hyper_parameter, sigma_x_hyper_parameter, sigma_a_hyper_parameter, True);
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