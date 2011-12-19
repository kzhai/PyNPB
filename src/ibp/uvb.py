"""
VariationalBayes for IBP
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import numpy, scipy;
import scipy.special;

"""
This is a python implementation of vanilla ibp, based on variational inference, with hyper parameter updating.

References:

"""
class VariationalBayes(object):
    """
    """
    def __init__(self,
                 snapshot_interval=10,
                 finite_mode=True,
                 model_likelihood_threshold=0.00001,
                 global_maximum_iteration=100):
        self._global_maximum_iteration = global_maximum_iteration;
        self._finite_mode = finite_mode;
        self._model_likelihood_threshold = model_likelihood_threshold;
        
        self._snapshot_interval = snapshot_interval;
    
    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self, data, truncation_level=10, alpha=1., sigma_f=1., sigma_n=1.):
        self._X = data;
        (self._N, self._D) = self._X.shape;

        self._K = truncation_level;
        
        self._alpha = alpha;
        self._sigma_f = sigma_f;
        self._sigma_n = sigma_n;
        
        # tau
        self._tau = numpy.ones((2, self._K));
        if self._finite_mode:
            self._tau[0, :] = self._alpha / self._K;
            self._tau += 0.5 * numpy.min(1., self._alpha/self._K) * (numpy.random.random(self._tau.shape) - 0.5);
        else:
            self._tau[0, :] = self._alpha;
            self._tau += 0.5 * numpy.min(1., self._alpha) * (numpy.random.random(self._tau.shape) - 0.5);
        # nu
        self._nu = numpy.random.random((self._N, self._K));
        # mean
        self._phi_mean = numpy.random.normal(0., 1., (self._K, self._D)) * 0.01;
        # covariance
        self._phi_cov = numpy.random.normal(0., 1., (self._K, self._D)) ** 2 * 0.1;

    """
    """
    def update_phi(self):
        nu_sum = numpy.sum(self._nu, axis=0)
        assert(len(nu_sum)==self._K);
        for k in numpy.random.permutation(xrange(self._K)):
            # we assume the covariance matrix is a diagonal matrix and we only store the diagonal terms
            self._phi_cov[k, :] = 1. / (1./(self._sigma_f * self._sigma_f) + nu_sum[k]/(self._sigma_n * self._sigma_n));

            phi_mean = numpy.dot(self._nu[:, k][:, numpy.newaxis].transpose(), self._X);
            assert(phi_mean.shape==(1, self._D));
            for n in xrange(self._N):
                phi_mean -= self._nu[n, k] * (numpy.dot(self._nu[n, :][numpy.newaxis, :], self._phi_mean) - numpy.dot(self._nu[n, k], self._phi_mean[k, :][numpy.newaxis, :]));
            assert(phi_mean.shape==(1, self._D));
            phi_mean /= self._sigma_n*self._sigma_n;
            self._phi_mean[k, :] = phi_mean * self._phi_cov[k, :];
            
        return

    """
    """
    def update_nu(self):
        var_theta_constant = self.compute_var_theta_constant();
        for k in numpy.random.permutation(xrange(self._K)):
            for n in numpy.random.permutation(xrange(self._N)):
                x_nu_phi = self._X[n, :][numpy.newaxis, :] - (numpy.dot(self._nu[n, :][numpy.newaxis, :], self._phi_mean) - numpy.dot(self._nu[n, k], self._phi_mean[k, :][numpy.newaxis, :]));
                assert(x_nu_phi.shape==(1, self._D));
                var_theta = var_theta_constant[k] + 1. / (self._sigma_n * self._sigma_n) * numpy.dot(self._phi_mean[k, :][numpy.newaxis, :], x_nu_phi.transpose());
                self._nu[n, k] = 1. / (1. + numpy.exp(-var_theta));
        return
    
    """
    """
    def update_tau(self):
        sum_nu = numpy.sum(self._nu, axis=0);
        if self._finite_mode:
            assert(len(sum_nu)==self._K);
            self._tau[0, :] = self._alpha / self._K + sum_nu;
            self._tau[1, :] = 1. + self._N - sum_nu;
        else:
            N_minus_sum_nu = self._N - sum_nu;
            psi_tau = scipy.special.psi(self._tau);
            assert(psi_tau.shape==(2, self._K));
            psi_sum_tau = scipy.special.psi(numpy.sum(self._tau, axis=0));
            assert(len(psi_tau.shape)==self._K);
            psi_tau0_cumsum = numpy.hstack([0, numpy.cumsum(psi_tau[0, :])]);
            assert(len(psi_tau0_cumsum)==self._K+1);
            psi_sum_cumsum = numpy.cumsum(psi_sum_tau);
            assert(len(psi_sum_cumsum)==self._K);
            exponent = psi_tau[1, :] + psi_tau_cumsum - psi_sum_cumsum;
            unnormalized = numpy.exp(exponent - numpy.max(exponent));
            assert(len(unnormalized)==self._K);
            for k in xrange(self._K):
                qs = numpy.zeros((self._K, self._K));
                for m in xrange(k, self._K):
                    qs[m, 0:m] = unnormalized[0:m] / numpy.sum(unnormalized[0:m]);
                
                self._tau[0, k] = numpy.sum(sum_nu[k:self._K]) + numpy.dot(N_minus_sum_nu[k+1:self._K], numpy.sum(qs[k+1:self._K, k+1:self._K], axis=1)) + self._alpha;
                self._tau[1, k] = numpy.dot(N_minus_sum_nu[k:self._K], qs[k:self._K, k]) + 1;
        return;
    
    """
    """
    def compute_var_theta_constant(self):
        var_theta_constant = - 0.5 / (self._sigma_n*self._sigma_n) * (numpy.sum(self._phi_cov, axis=1) + numpy.sum(self._phi_mean*self._phi_mean, axis=1));
        assert(len(var_theta_constant)==self._K);
        if self._finite_mode:
            var_theta_constant += scipy.special.psi(self._tau[0, :]) - scipy.special.psi(self._tau[1, :]);
        else:
            for k in xrange(self._K):
                var_theta_constant[k] += numpy.sum(scipy.special.psi(self._tau[0, 0:k]) - scipy.special.psi(numpy.sum(self._tau[:, 0:k])));
                var_theta_constant[k] -= self.compute_expected_pzk0_qjensen(k);
        return var_theta_constant;

    """
    """
    def compute_expected_pzk0_qjensen(self, k):
        assert(k>=0 and k<self._K);
        tau = self._tau[:, 0:k+1];
        assert(tau.shape==(2, k));
        psi_tau = scipy.special.psi(tau);
        assert(psi_tau.shape==(2, k));
        psi_sum_tau = scipy.special.psi(numpy.sum(tau, axis=0));
        assert(len(psi_tau.shape)==k);
        psi_tau0_cumsum = numpy.hstack([0, numpy.cumsum(psi_tau[0, :])]);
        assert(len(psi_tau0_cumsum)==k+1);
        psi_sum_cumsum = numpy.cumsum(psi_sum_tau);
        assert(len(psi_sum_cumsum)==k);
        tmp = psi_tau[1, :] + psi_tau0_cumsumd[0:k] - psi_sum_cumsum;
        assert(len(tmp)==k);
        q = numpy.exp(tmp - numpy.max(tmp));
        assert(len(q)==k);
        q = q / numpy.sum(q);
        assert(len(q)==k);

        # compute the lower bound
        lower_bound = numpy.sum(q * (tmp - numpy.log(q)));
        
        return lower_bound

    """
    """
    def learning(self, iteration=0, directory="../../output/tmp-output"):
        if iteration <= 0:
            iteration = self._global_maximum_iteration;
        
        for i in xrange(iteration):
            self.update_nu();
            self.update_phi();
            self.update_tau();

        print "learning finished..."
    
    """
    compute the evidence lower bound (i.e. elb) in log scale
    """
    def elb(self):
        log_likelihood = numpy.zeros(5);
        
        psi_tau = scipy.special.psi(self._tau);
        assert(psi_tau.shape==(2, self._K));
        psi_sum_tau = scipy.special.psi(numpy.sum(self._tau, axis=1)[:, numpy.newaxis]);
        assert(psi_sum_tau.shape==(1, self._K));

        # entropy of the proposed distribution
        lngamma_tau = scipy.special.lngamma(self._tau);
        assert(lngamma_tau.shape==(2, self._K));
        lngamma_sum_tau = scipy.special.lngamma(numpy.sum(self._tau, axis=1)[:, numpy.newaxis]);
        assert(lngamma_sum_tau.shape==(1, self._K));
        
        log_likelihood[4] = numpy.sum(lngamma_tau[0, :] + lngamma_tau[1, :] - lngamma_sum_tau);
        log_likelihood[4] -= numpy.sum((self._tau[0, :]-1) * psi_tau[0, :] + (self._tau[1, :]-1) * psi_tau[1, :]);
        log_likelihood[4] += numpy.sum((self._tau[0, :] + self._tau[1, :] - 2) * psi_sum_tau);
        
        assert(numpy.all(self._phi_cov>0));
        assert(numpy.all(self._nu>0));
        log_likelihood[4] += 0.5 * numpy.sum((self._D * numpy.log(2 * numpy.pi * numpy.e) + numpy.log(numpy.sqrt(numpy.sum(self._phi_cov * self._phi_cov, axis=1)))));
        log_likelihood[4] -= numpy.sum(self._nu * numpy.log(self._nu) + (1.-self._nu) * numpy.log(1.-self._nu)));

        if self._finite_mode:
            log_likelihood[0] = self._K * numpy.log(self._alpha / self._K) + (self._alpha / self._K) * numpy.sum(psi_tau[0, :] - psi_sum_tau);
            
            log_likelihood[1] = numpy.sum(self._nu * psi_tau[0, :]) + numpy.sum((1-self._nu) * psi_tau[1, :]) - self._N * numpy.sum(psi_sum_tau);
            
            log_likelihood[2] = - 0.5 * self._K * self._D * numpy.log(2 * numpy.pi * self._sigma_f * self._sigma_f);
            log_likelihood[2] -= 0.5 / (self._sigma_f * self._sigma_f) * (numpy.sum(self._phi_cov) + numpy.sum(self._phi_mean * self._phi_mean));
            
            tmp_log_likelihood = numpy.sum(self._X * self._X) - 2 * numpy.sum(numpy.dot(self._nu, self._phi_mean) * self._X)
            log_likelihood[3] = - 0.5 * self._K * self._D * numpy.log(2 * numpy.pi * self._sigma_n * self._sigma_n);
            log_likelihood[3] -= 0.5 / (self._sigma_n * self._sigma_n) * (numpy.sum(self._X * self._X)
                                                                          - 2 * numpy.sum(numpy.dot(self._nu, self._phi_mean) * self._X);
        else:
            return;

if __name__ == "__main__":
    import scipy.io;
    
    # load the data from the matrix
    mat_vals = scipy.io.loadmat('../../data/cambridge-bars/block_image_set.mat');
    true_weights = mat_vals['trueWeights']
    features = mat_vals['features']
    data = mat_vals['data']
    
    # initialize the model
    ibp = VariationalBayes();

    ibp._initialize(data[1:100, :]);
    
    ibp.learning(100);
    
    # If matplotlib is installed, plot ground truth vs learned factors
    import matplotlib.pyplot as P
    from util.scaled_image import scaledimage
    
    # Intensity plots of
    # -ground truth factor-feature weights (top)
    # -learned factor-feature weights (bottom)
    K = max(len(true_weights), len(ibp._phi_mean))
    (fig, subaxes) = P.subplots(2, K)
    for sa in subaxes.flatten():
        sa.set_visible(False)
    fig.suptitle('Ground truth (top) vs learned factors (bottom)')
    for (idx, trueFactor) in enumerate(true_weights):
        ax = subaxes[0, idx]
        ax.set_visible(True)
        scaledimage(trueFactor.reshape(6, 6),
                    pixwidth=3, ax=ax)
    for (idx, learnedFactor) in enumerate(ibp._phi_mean):
        ax = subaxes[1, idx]
        scaledimage(learnedFactor.reshape(6, 6),
                    pixwidth=3, ax=ax)
        ax.set_visible(True)
    P.show()