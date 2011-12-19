"""
VariationalBayes
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import abc;
import math, random;
import numpy, scipy;

"""
"""
class VariationalBayes(object):
    __metaclass__ = abc.ABCMeta;
    """
    @param num_topics: the number of topics
    @param data: a defaultdict(dict) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self, data, num_topics=10):
        # initialize the total number of topics.
        self._K = num_topics
        
        # initialize a K-dimensional vector, valued at 1/K.
        self._alpha = numpy.random.random((1, self._K)) / self._K;

        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        from util.input_parser import dict_list_2_dict_freqdist
        data = dict_list_2_dict_freqdist(data);
        self._data = data
        
        # initialize the size of the collection, i.e., total number of documents.
        self._D = len(self._data)
        
        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._vocab = []
        for token_list in data.values():
            self._vocab += token_list
        self._vocab = list(set(self._vocab))
        
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        self._V = len(self._vocab)
        
        # initialize a D-by-K matrix gamma, valued at N_d/K
        self._gamma = numpy.tile(self._alpha + 1.0 * self._V / self._K, (self._D, 1));
        
        # initialize a V-by-K matrix beta, valued at 1/V, subject to the sum over every row is 1
        self._log_beta = 1.0 / self._V + numpy.random.random((self._V, self._K));
        self._log_beta = self._log_beta / numpy.sum(self._log_beta, axis=0)[numpy.newaxis, :];
        self._log_beta = numpy.log(self._log_beta);

    """
    @param alpha_vector: a dict data type represents dirichlet prior, indexed by topic_id
    @param alpha_sufficient_statistics: a dict data type represents alpha sufficient statistics for alpha updating, indexed by topic_id
    @param rho: a step size adjustment factor, set to 1 if vanilla lda 
    """
    def update_alpha(self, alpha_sufficient_statistics, rho):
        assert(alpha_sufficient_statistics.shape == (1, self._K));        
        alpha_update = self._alpha;
        
        decay = 0;
        for alpha_iteration in xrange(self._alpha_maximum_iteration):
            alpha_sum = numpy.sum(self._alpha);
            alpha_gradient = self._D * (scipy.special.psi(alpha_sum) - scipy.special.psi(self._alpha)) + alpha_sufficient_statistics;
            alpha_hessian = -self._D * scipy.special.polygamma(1, self._alpha);

            if numpy.any(numpy.isinf(alpha_gradient)) or numpy.any(numpy.isnan(alpha_gradient)):
                print "illegal alpha gradient vector", alpha_gradient

            sum_g_h = numpy.sum(alpha_gradient / alpha_hessian);
            sum_1_h = 1.0 / alpha_hessian;

            z = self._D * scipy.special.polygamma(1, alpha_sum);
            c = sum_g_h / (1.0 / z + sum_1_h);

            # update the alpha vector
            while True:
                singular_hessian = False

                step_size = numpy.power(self._alpha_update_decay_factor, decay) * (alpha_gradient - c) / alpha_hessian;
                step_size *= rho;
                #print "step size is", step_size
                assert(self._alpha.shape == step_size.shape);
                
                if numpy.any(self._alpha <= step_size):
                    singular_hessian = True
                else:
                    alpha_update = self._alpha - step_size;
                
                if singular_hessian:
                    decay += 1;
                    if decay > self._alpha_maximum_decay:
                        break;
                else:
                    break;
                
            # compute the alpha sum
            # check the alpha converge criteria
            mean_change = numpy.mean(abs(alpha_update - self._alpha));
            self._alpha = alpha_update;
            if mean_change <= self._alpha_converge_threshold:
                break;

        return

    """
    """
    def print_topics(self, term_mapping, top_words=10):
        input = open(term_mapping);
        vocab = {};
        i = 0;
        for line in input:
            vocab[i] = line.strip();
            i += 1;

        if top_words >= self._V:
            sorted_beta = numpy.zeros((1, self._K)) - numpy.log(self._V);
        else:
            sorted_beta = numpy.sort(self._log_beta, axis=0);
            sorted_beta = sorted_beta[-top_words, :][numpy.newaxis, :];

        #print sorted_beta;
        
        #display = self._log_beta > -numpy.log(self._V);
        #assert(display.shape==(self._V, self._K));
        for k in xrange(self._K):
            display = self._log_beta[:, [k]] >= sorted_beta[:, k];
            assert(display.shape == (self._V, 1));
            output_str = str(k) + ": ";
            for v in xrange(self._V):
                if display[v, :]:
                    output_str += vocab[v] + "\t";
            print output_str
