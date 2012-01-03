import numpy, scipy;
import scipy.special;

"""
@author: Ke Zhai (zhaike@cs.umd.edu)

This is a python implementation of vanilla Indian Buffet process, based on variational inference.

References:
[1] Finale Doshi-Velez, Kurt T. Miller, Jurgen Van Gael, and Yee Whye Teh, Variational Inference for the Indian Buffet Process, Artificial Intelligence and Statistics (AISTATS), 2009.
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
    @param data: a N-by-D matrix, representing N images, each of which contains D pixel
    @param truncation_level: truncation level for variational inference indian buffet process
    @param alpha: hyper-parameter defining the indian buffet process
    @param sigma_a: feature sigma
    @param sigma_x: data sigma
    """
    def _initialize(self, data, truncation_level=5, alpha=1., sigma_a=1., sigma_x=1.):
        self._X = data;
        (self._N, self._D) = self._X.shape;

        self._K = truncation_level;
        
        self._alpha = alpha;
        self._sigma_a = sigma_a;
        self._sigma_x = sigma_x;
        
        # tau
        self._tau = numpy.ones((2, self._K));
        if self._finite_mode:
            self._tau[0, :] = self._alpha / self._K;
            self._tau += 0.5 * numpy.min(1., self._alpha / self._K) * (numpy.random.random(self._tau.shape) - 0.5);
        else:
            self._tau[0, :] = self._alpha;
            self._tau += 0.5 * numpy.min(1., self._alpha) * (numpy.random.random(self._tau.shape) - 0.5);
        assert(self._tau.shape == (2, self._K));

        # nu
        self._nu = numpy.random.random((self._N, self._K));
        assert(self._nu.shape == (self._N, self._K));

        # phi
        self._phi_mean = numpy.random.normal(0., 1., (self._K, self._D)) * 0.01;
        self._phi_cov = numpy.random.normal(0., 1., (self._K, self._D)) ** 2 * 0.1;
        assert(self._phi_mean.shape == (self._K, self._D));
        assert(self._phi_cov.shape == (self._K, self._D));

    """
    """
    def update_phi(self):
        nu_sum_n = numpy.sum(self._nu, axis=0);
        assert(len(nu_sum_n) == self._K);
        nu_sum_k = numpy.sum(self._nu, axis=1);
        assert(len(nu_sum_k) == self._N);
        for k in numpy.random.permutation(xrange(self._K)):
            # we assume the covariance matrix is a diagonal matrix and we only store the diagonal terms
            self._phi_cov[k, :] = 1. / (1. / (self._sigma_a * self._sigma_a) + nu_sum_n[k] / (self._sigma_x * self._sigma_x));

            tmp_nu = numpy.delete(self._nu, k, 1);
            assert(tmp_nu.shape == (self._N, self._K - 1));
            tmp_phi_mean = numpy.delete(self._phi_mean, k, 0);
            assert(tmp_phi_mean.shape == (self._K - 1, self._D));
            phi_mean = numpy.dot(self._X.transpose(), self._nu[:, k][:, numpy.newaxis]).transpose();
            for n in xrange(self._N):
                phi_mean -= self._nu[n, k] * numpy.sum(tmp_nu[n, :] * tmp_phi_mean.transpose(), axis=1)[:, numpy.newaxis].transpose();
            assert(phi_mean.shape == (1, self._D));
            phi_mean /= self._sigma_x * self._sigma_x;
            self._phi_mean[k, :] = phi_mean * self._phi_cov[k, :];
            
        return

    """
    """
    def update_nu(self):
        var_theta_constant = self.compute_var_theta_constant();
        for k in numpy.random.permutation(xrange(self._K)):
            tmp_nu = numpy.delete(self._nu, k, 1);
            assert(tmp_nu.shape == (self._N, self._K - 1));
            tmp_phi_mean = numpy.delete(self._phi_mean, k, 0);
            assert(tmp_phi_mean.shape == (self._K - 1, self._D));
            
            for n in numpy.random.permutation(xrange(self._N)):
                x_nu_phi = self._X[n, :][numpy.newaxis, :] - numpy.sum(tmp_nu[n, :] * tmp_phi_mean.transpose(), axis=1)[:, numpy.newaxis].transpose();
                assert(x_nu_phi.shape == (1, self._D));
                var_theta = var_theta_constant[k] + 1. / (self._sigma_x * self._sigma_x) * numpy.dot(self._phi_mean[k, :][numpy.newaxis, :], x_nu_phi.transpose());
                self._nu[n, k] = 1. / (1. + numpy.exp(-var_theta));
                
        return;
    
    """
    """
    def update_tau(self):
        sum_nu = numpy.sum(self._nu, axis=0);
        assert(len(sum_nu) == self._K);
        if self._finite_mode:
            assert(len(sum_nu) == self._K);
            self._tau[0, :] = self._alpha / self._K + sum_nu;
            self._tau[1, :] = 1. + self._N - sum_nu;
        else:
            N_minus_sum_nu = self._N - sum_nu;
            for k in xrange(self._K):
                psi_tau = scipy.special.psi(self._tau);
                assert(psi_tau.shape == (2, self._K));
                
                psi_sum_tau = scipy.special.psi(numpy.sum(self._tau, axis=0));
                assert(len(psi_sum_tau) == self._K);
                
                psi_tau0_cumsum = numpy.hstack([0, numpy.cumsum(psi_tau[0, :-1])]);
                assert(len(psi_tau0_cumsum) == self._K);
                
                psi_sum_cumsum = numpy.cumsum(psi_sum_tau);
                assert(len(psi_sum_cumsum) == self._K);
                
                exponent = psi_tau[1, :] + psi_tau0_cumsum - psi_sum_cumsum;
                unnormalized = numpy.exp(exponent - numpy.max(exponent));
                assert(len(unnormalized) == self._K);
                
                qs = numpy.zeros((self._K, self._K));
                for m in xrange(k, self._K):
                    qs[m, 0:m + 1] = unnormalized[0:m + 1] / numpy.sum(unnormalized[0:m + 1]);
                
                self._tau[0, k] = numpy.sum(sum_nu[k:self._K]) + numpy.dot(N_minus_sum_nu[k + 1:self._K], numpy.sum(qs[k + 1:self._K, k + 1:self._K], axis=1)) + self._alpha;
                self._tau[1, k] = numpy.dot(N_minus_sum_nu[k:self._K], qs[k:self._K, k]) + 1;
        return;
    
    """
    """
    def compute_var_theta_constant(self):
        var_theta_constant = -0.5 / (self._sigma_x * self._sigma_x) * ((numpy.sum(self._phi_cov, axis=1) + numpy.sum(self._phi_mean * self._phi_mean, axis=1)));
        assert(len(var_theta_constant) == self._K);
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
        assert(k >= 0 and k < self._K);
        tau = self._tau[:, 0:k + 1];
        assert(tau.shape == (2, k + 1));

        psi_tau = scipy.special.psi(tau);
        assert(psi_tau.shape == (2, k + 1));

        psi_sum_tau = scipy.special.psi(numpy.sum(tau, axis=0));
        assert(len(psi_sum_tau) == k + 1);

        psi_tau0_cumsum = numpy.hstack([0, numpy.cumsum(psi_tau[0, :-1])]);
        assert(len(psi_tau0_cumsum) == k + 1);

        psi_sum_cumsum = numpy.cumsum(psi_sum_tau);
        assert(len(psi_sum_cumsum) == k + 1);

        tmp = psi_tau[1, :] + psi_tau0_cumsum - psi_sum_cumsum;
        assert(len(tmp) == k + 1);
        
        q = numpy.exp(tmp - numpy.max(tmp));
        assert(len(q) == k + 1);
        
        q = q / numpy.sum(q);
        assert(len(q) == k + 1);

        # compute the lower bound
        lower_bound = numpy.sum(q * (tmp - numpy.log(q)));
        
        return lower_bound

    """
    """
    def learning(self, iteration=0, directory="../../output/tmp-output"):
        if iteration <= 0:
            iteration = self._global_maximum_iteration;

        print "likelihood is", self.velb();
        
        for i in xrange(iteration):
            self.update_tau();
            self.update_phi();
            self.update_nu();
            print "likelihood is", self.velb()
            
        print "learning finished..."
    
    """
    compute the variational evidence lower bound (i.e. velb) in log scale
    """
    def velb(self):
        log_likelihood = numpy.zeros(5);
        
        psi_tau = scipy.special.psi(self._tau);
        assert(psi_tau.shape == (2, self._K));
        psi_sum_tau = scipy.special.psi(numpy.sum(self._tau, axis=0)[numpy.newaxis, :]);
        assert(psi_sum_tau.shape == (1, self._K));

        if self._finite_mode:
            # compute the probability of feature
            log_likelihood[0] = self._K * numpy.log(self._alpha / self._K) + (self._alpha / self._K - 1.) * numpy.sum(psi_tau[0, :] - psi_sum_tau);
            # compute the probability of feature statistics
            log_likelihood[1] = numpy.sum(self._nu * psi_tau[0, :]) + numpy.sum((1. - self._nu) * psi_tau[1, :]) - self._N * numpy.sum(psi_sum_tau);
        else:
            # compute the probability of feature
            log_likelihood[0] = self._K * numpy.log(self._alpha) + (self._alpha - 1.) * numpy.sum(psi_tau[0, :] - psi_sum_tau);
            # compute the probability of feature statistics
            for k in xrange(self._K):
                log_likelihood[1] += numpy.sum(self._nu[:, k]) * numpy.sum(psi_tau[0, :k + 1] - psi_sum_tau[0, :k + 1]);
                log_likelihood[1] += numpy.dot((self._N - numpy.sum(self._nu[:, k])), self.compute_expected_pzk0_qjensen(k));
            
        # compute the probability of feature distribution
        log_likelihood[2] = -0.5 * self._K * self._D * numpy.log(2 * numpy.pi * self._sigma_a * self._sigma_a);
        log_likelihood[2] -= 0.5 / (self._sigma_a * self._sigma_a) * (numpy.sum(self._phi_cov) + numpy.sum(self._phi_mean * self._phi_mean));
        
        # compute the probability of data likelihood
        tmp_log_likelihood = numpy.sum(self._X * self._X) - 2 * numpy.sum(self._nu * numpy.dot(self._X, self._phi_mean.transpose()));
        tmp_1 = numpy.dot(numpy.ones((self._N, self._D)), (self._phi_cov + self._phi_mean ** 2).transpose());
        tmp_log_likelihood += numpy.sum(self._nu * tmp_1);
        tmp_1 = numpy.dot(self._nu, self._phi_mean);
        tmp_2 = numpy.sum(numpy.dot(self._nu ** 2, self._phi_mean ** 2));
        tmp_log_likelihood += numpy.sum(tmp_1 * tmp_1) - numpy.sum(tmp_2);
        
        log_likelihood[3] = -0.5 * self._N * self._D * numpy.log(2 * numpy.pi * self._sigma_x * self._sigma_x);
        log_likelihood[3] -= 0.5 / (self._sigma_x * self._sigma_x) * tmp_log_likelihood
        
        # entropy of the proposed distribution
        lngamma_tau = scipy.special.gammaln(self._tau);
        assert(lngamma_tau.shape == (2, self._K));
        lngamma_sum_tau = scipy.special.gammaln(numpy.sum(self._tau, axis=0)[numpy.newaxis, :]);
        assert(lngamma_sum_tau.shape == (1, self._K));
        
        # compute the entropy of the distribution
        log_likelihood[4] = numpy.sum(lngamma_tau[0, :] + lngamma_tau[1, :] - lngamma_sum_tau);
        log_likelihood[4] -= numpy.sum((self._tau[0, :] - 1) * psi_tau[0, :] + (self._tau[1, :] - 1) * psi_tau[1, :]);
        log_likelihood[4] += numpy.sum((self._tau[0, :] + self._tau[1, :] - 2) * psi_sum_tau);
        
        assert(numpy.all(self._phi_cov > 0));
        assert(numpy.all(self._nu >= 0) and numpy.all(self._nu <= 1));
        log_likelihood[4] += 0.5 * self._K * self._D * numpy.log(2 * numpy.pi * numpy.e);
        log_likelihood[4] += 0.5 * numpy.sum(numpy.log(self._phi_cov));
        #log_likelihood[4] += 0.5 * numpy.log(numpy.sqrt(numpy.sum(self._phi_cov * self._phi_cov, axis=1)));
        log_likelihood[4] -= numpy.sum(self._nu * numpy.log(self._nu) + (1. - self._nu) * numpy.log(1. - self._nu));
        
        return numpy.sum(log_likelihood);

if __name__ == "__main__":
    import scipy.io;
    import numpy;
    
    # load the data from the matrix
    mat_vals = scipy.io.loadmat('../../data/cambridge-bars/block_image_set.mat');
    true_weights = mat_vals['trueWeights']
    features = mat_vals['features']
    data = mat_vals['data']
    
    # initialize the model
    ibp = VariationalBayes(10, False, 0.00001, 100)
    #ibp._initialize(data[1:10, :], 4, true_weights, 1., 1., 1.);
    ibp._initialize(data[1:10, :])
    ibp.learning(10);
    
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
