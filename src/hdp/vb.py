import numpy, scipy;
import scipy.special;

"""
@author: Ke Zhai (zhaike@cs.umd.edu)

Implements variational bayes for the hierarchical Dirichlet process (HDP).
"""
class VariationalBayes(object):
    """
    """
    def __init__(self,
                 snapshot_interval=10,
                 finite_mode=True,
                 lambda=None,
                 model_likelihood_threshold=0.00001,
                 global_maximum_iteration=100):
        self._global_maximum_iteration = global_maximum_iteration;
        self._finite_mode = finite_mode;
        assert(lambda == None or type(lambda) == tuple);
        self._lambda = lambda;
        self._model_likelihood_threshold = model_likelihood_threshold;
        
        self._snapshot_interval = snapshot_interval;
    
    """
    @param data:
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self, data, K=5):
        # initialize the total number of topics.
        self._K = K;

        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        data = dict_list_2_dict_freqdist(data);
        self._data = data
        print data

        # initialize the size of the collection, i.e., total number of documents.
        self._D = len(self._data)

        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._vocab = []
        for token_list in data.values():
            self._vocab += token_list
        self._vocab = list(set(self._vocab))

        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        self._V = len(self._vocab)

        # initialize a V-by-K matrix phi, valued at random.
        self._phi = numpy.random.random(self._V, self._K);
        assert(self._phi.shape == (self._V, self._K));
        
        # initialize a (K-1)-by-2 matrix gamma, valued at random.
        self._gamma = numpy.random.random((self._K, 2));
        assert(self._gamma.shape == (self._K, 2));
        
        # initialize a K-by-2 matrix beta, valued at random.
        self._tau = numpy.random.random((self._K, 2));
        assert(self._tau.shape == (self._K, 2));

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
    def update_gamma(self):
        self._gamma[0, :] = 1 + numpy.sum(self._phi, axis=0);
        
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
    

"""
some utility functions
"""
def import_monolingual_data(input_file):
    import codecs
    input = codecs.open(input_file, mode="r", encoding="utf-8")
    
    doc_count = 0
    docs = {}
    
    for line in input:
        line = line.strip().lower();

        contents = line.split("\t");
        assert(len(contents) == 2);
        docs[int(contents[0])] = [int(item) for item in contents[1].split()];

        doc_count += 1
        if doc_count % 10000 == 0:
            print "successfully import " + str(doc_count) + " documents..."

    print "successfully import all documents..."
    return docs
"""
this method convert a corpus from dict(list) to defaultdict(FreqDist) format, similar words are grouped
@return: a defaultdict(FreqDist) data type, first indexed by the document id, then indexed by the unique tokens
@param corpus: a dict data type, indexed by document id, corresponding value is a list of words (not necessarily unique from each other)
"""
def dict_list_2_dict_freqdist(corpus):
    from collections import defaultdict;
    from nltk.probability import FreqDist;

    docs = defaultdict(dict)
    
    for doc in corpus.keys():
        docs[doc] = FreqDist(corpus[doc])
    
    return docs

if __name__ == "__main__":
    temp_directory = "../../data/test/";
    #temp_directory = "../../data/de-news/en/corpus-3/";
    data = import_monolingual_data(temp_directory + "doc.dat");

    vb = VariationalBayes();
    vb._initialize(data);
