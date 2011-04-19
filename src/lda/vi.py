# Author: Ke Zhai
# Email: zhaike@cs.umd.edu

from collections import defaultdict;
from math import log, exp, fabs, pow, isnan, isinf;
from util.log_math import log_add;
from random import random;
from copy import deepcopy;
from scipy.special import psi, gammaln, polygamma;

# this is a python implementation of lda based on variational inference.
# the algorithm follows the documentataion in Blei's paper "Latent Dirichlet Allocation"
class VariationalInference(object):
    def __init__(self):
    #def __init__(self, gamma_converge=0.000001, gamma_maximum_iteration=400, alpha_converge=0.000001, alpha maximum_iteration=100, em_maximum_iteration = 5, em_converge = 0.00001):
        # initialize the iteration parameters
        self._alpha_update_decay_factor = 0.9
        self._alpha_maximum_decay = 10
        
        self._gamma_converge = 0.000001
        self._gamma_maximum_iteration = 400
        
        self._alpha_converge = 0.000001
        self._alpha_maximum_iteration = 100
        
        self._maximum_iteration = 100
        self._converge = 0.00001
        
    """
    @param num_topics: the number of topics
    @param data: a defaultdict(FreqDist) data type, first indexed by doc id then indexed by term id
    take note: words are not terms, they are repeatable and thus might be not unique
    """
    def _initialize(self, data, num_topics=10):
        # initialize the total number of topics.
        self._K = num_topics
        
        # initialize a K-dimensional vector, valued at 1/K.
        self._alpha = {}
        for k in range(self._K):
            self._alpha[k] = random() / self._K
        #print self._alpha

        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        self._data = data
        
        # initialize the size of the collection, i.e., total number of documents.
        self._D = len(self._data)
        
        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._vocab = []
        for token_list in data.values():
            self._vocab += token_list
        #print len(self._vocab)
        #self._vocab = list(set(self._vocab))
        self._vocab = set(self._vocab)
        
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        self._V = len(self._vocab)
        #print self._V
        
        # initialize a D-by-K matrix gamma, valued at N_d/K
        self._gamma = defaultdict(dict)
        for d in self._data.keys():
            temp = {}
            for k in range(self._K):
                temp[k] = self._alpha[k] + 1.0 * self._V / self._K;
            self._gamma[d] = temp
        #print self._gamma
        
        # initialize a V-by-K matrix beta, valued at 1/V, subject to the sum over every row is 1
        self._beta = defaultdict(dict)
        for v in self._vocab:
            temp = {}
            for k in range(self._K):
                temp[k] = log(1.0 / self._V + random())
            self._beta[v] = temp
        #print self._beta

    def inference(self):
        # initialize the likelihood factor
        likelihood_alpha = 0.0
        likelihood_gamma = 0.0
        likelihood_phi = 0.0
        
        # initialize the computational parameters
        alpha_sum = 0.0;
        for k in range(self._K):
            alpha_sum += self._alpha[k]
            likelihood_alpha -= gammaln(self._alpha[k])
        
        likelihood_alpha += gammaln(alpha_sum)
        likelihood_alpha *= self._D
        
        # initialize a V-by-K matrix beta contribution
        beta = defaultdict(dict)
        beta_normalize_factor = {}
        
        # initialize a K-dimensional row vector to compute the normalizing factor
        sums = {}
        gamma_token = {}
        for k in range(self._K):
            sums[k] = 0.0
            gamma_token[k] = 0.0;
            
        # initialize a V-by-K matrix phi contribution
        phi_table = defaultdict(dict)
        
        # initialize a K vector alpha sufficient statistics
        alpha_sufficient_statistics = {}
        
        # iterate over all documents
        for doc in self._data.keys():
            
            # compute the total number of words
            total_word_count = self._data[doc].N()
                
            # initialize the sum of gamma values 
            sum_gamma = 0.0
            
            # initialize gamma for this document
            gamma = {}
            for k in range(self._K):
                gamma[k] = self._alpha[k] + 1.0*total_word_count/self._K
                sum_gamma += gamma[k]

            # iterate till convergence
            for gamma_iteration in range(self._gamma_maximum_iteration):
                # initialize gamma update for this document
                gamma_update = {}
                for k in range(self._K):
                    gamma_update[k] = log(self._alpha[k])
                
                # update phi vector, i.e., pseudo-counts for word-topic assignment (beta vector)
                [gamma_update, phi_table, likelihood_phi_temp] = self.update_phi(self._data[doc], phi_table, self._beta, gamma, gamma_update)
                
                keep_going = False
                for k in range(self._K):
                    if abs((gamma_update[k] - gamma[k]) / gamma[k]) > self._gamma_converge:
                        keep_going = True
                        break

                gamma = gamma_update

                if not keep_going:
                    break
                
            self._gamma[doc] = gamma
            
            sum_gamma = 0.0
            for k in range(self._K):
                likelihood_gamma += gammaln(gamma[k])
                sum_gamma += gamma[k]
            likelihood_gamma -= gammaln(sum_gamma)
            
            likelihood_phi += likelihood_phi_temp
                
            [beta, beta_normalize_factor] = self.update_beta(self._data[doc], phi_table, beta_normalize_factor, beta)

            for k in range(self._K):
                if k not in alpha_sufficient_statistics.keys():
                    alpha_sufficient_statistics[k] = psi(gamma[k]) - psi(sum_gamma)
                else:
                    alpha_sufficient_statistics[k] += psi(gamma[k]) - psi(sum_gamma)
            
        self._beta = self.normalize_beta(beta_normalize_factor, beta)
        
        self._alpha = self.update_alpha(self._alpha, alpha_sufficient_statistics)
        
        likelihood = likelihood_alpha + likelihood_gamma + likelihood_phi

        return likelihood

    """
    @param doc: a dict data type represents the content of a document, indexed by term_id
    @param phi_table: a defaultdict(dict) data type represents the phi matrix, in size of V-by-K, first indexed by term_id, then indexed by topic_id
    @param beta: a defaultdict(dict) data type represents the beta mtarix, in size of V-by-K, first indexed by term_id, then indexed by topic_id
    @param gamma: a defaultdict(dict) data type represents the gamma matrix, in size of D-by-K, first indexed by doc_id, then indexed by topic_id
    @param alpha: a dict data type represents the alpha vector, in size of K, indexed by topic_id
    doc, beta, gamma and alpha value will not be modified, however, phi_table will be updated during this function
    """
    def update_phi(self, doc, phi_table, beta, gamma, gamma_update):
        # initialize 
        phi_table.clear()
        likelihood_phi = 0.0
        
        # iterate over all terms in the particular document
        for term in doc.keys():
            term_counts = doc[term]
            
            phi_normalize_factor = 0.0
             
            if term not in phi_table.keys():
                phi_table[term] = {}
             
            # iterate over all topics
            for k in range(self._K):
                if k not in phi_table[term].keys():
                    phi_table[term][k] = 0
                
                phi_table[term][k] = beta[term][k] + psi(gamma[k])
                #phi[k] = self._beta[term][k] + sp.special.psi(self._gamma[doc][k])
                # this term is constant, could be ignored after normalization
                #- sp.special.psi(gamma_sum)
                if(k==0):
                    phi_normalize_factor = phi_table[term][k]
                else:
                    phi_normalize_factor = log_add(phi_normalize_factor, phi_table[term][k])
                        
            # sums the K-dimensional row vector phi
            for k in range(self._K):
                # normalize the term
                phi_table[term][k] -= phi_normalize_factor
                likelihood_phi += term_counts * exp(phi_table[term][k]) * (beta[term][k] - phi_table[term][k])
                phi_table[term][k] += log(term_counts)
                
                # update the K-dimensional row vector gamma[doc]
                gamma_update[k] = log_add(gamma_update[k], phi_table[term][k])
                         
        sum_gamma = 0.0
        # gamma update is in log scale, remember?
        for k in range(self._K):
            gamma_update[k] = exp(gamma_update[k])
            sum_gamma += gamma_update[k]
        
        return gamma_update, phi_table, likelihood_phi
    
    """
    @deprecated: please use update_phi instead, which includes a trick to compute the likelihood_phi
    @param doc: a dict data type represents the content of a document, indexed by term_id
    @param phi_table: a defaultdict(dict) data type represents the phi matrix, in size of V-by-K, first indexed by term_id, then indexed by topic_id
    @param beta: a defaultdict(dict) data type represents the beta mtarix, in size of V-by-K, first indexed by term_id, then indexed by topic_id
    @param gamma: a defaultdict(dict) data type represents the gamma matrix, in size of D-by-K, first indexed by doc_id, then indexed by topic_id
    @param alpha: a dict data type represents the alpha vector, in size of K, indexed by topic_id
    doc, beta, gamma and alpha value will not be modified, however, phi_table will be updated during this function
    """
    def update_gamma(self, doc, phi_table, beta, gamma, gamma_update):
        # initialize 
        clear_phi_table = True
        phi_table.clear()

        phi_sum = {}
        phi_weighted_sum = {}
        likelihood_phi = 0.0
        
        # iterate over all terms in the particular document
        for term in doc.keys():
            term_counts = doc[term]
            
            phi_normalize_factor = 0.0
             
            if term not in phi_table.keys():
                phi_table[term] = {}
             
            # iterate over all topics
            for k in range(self._K):
                if k not in phi_table[term].keys():
                    phi_table[term][k] = 0
                
                phi_table[term][k] = beta[term][k] + psi(gamma[k])
                #phi[k] = self._beta[term][k] + sp.special.psi(self._gamma[doc][k])
                # this term is constant, could be ignored after normalization
                #- sp.special.psi(gamma_sum)
                if(k==0):
                    phi_normalize_factor = phi_table[term][k]
                else:
                    phi_normalize_factor = log_add(phi_normalize_factor, phi_table[term][k])
                        
            # sums the K-dimensional row vector phi
            for k in range(self._K):
                # normalize the term
                phi_table[term][k] -= phi_normalize_factor
                
                #print term, k, phi_table[term][k], phi_sum[k]
                
                if clear_phi_table:
                    phi_sum[k] = phi_table[term][k]
                else:
                    phi_sum[k] = log_add(phi_sum[k], phi_table[term][k])

                likelihood_phi += exp(phi_table[term][k]) * (phi_table[term][k] + term_counts * beta[term][k])
                phi_table[term][k] += log(term_counts)
                
                if clear_phi_table :
                    phi_weighted_sum[k] = phi_table[term][k]
                else:
                    phi_weighted_sum[k] = log_add(phi_weighted_sum[k], phi_table[term][k])
                
                # update the K-dimensional row vector gamma[doc]
                gamma_update[k] = log_add(gamma_update[k], phi_table[term][k])
                
            clear_phi_table = False
         
        sum_gamma = 0.0
        # gamma update is in log scale, remember?
        for k in range(self._K):
            gamma_update[k] = exp(gamma_update[k])
            sum_gamma += gamma_update[k]
                
        for k in range(self._K):
            likelihood_phi += (exp(phi_sum[k]) - exp(phi_weighted_sum[k])) * (psi(gamma_update[k]) - psi(sum_gamma))
        
        return gamma_update, phi_table, likelihood_phi
        
    """
    @param doc: a dict data type represents the content of a document, indexed by term_id
    @param phi_table: a defaultdict(dict) data type represents the phi matrix, in size of V-by-K, first indexed by term_id, then indexed by topic_id
    @param beta_normalize_factor: a dict data type represents the beta normalize factor
    @param update_beta: a defaultdict(dict) data type represents the beta matrix, in size of V-by-K, first indexed by term_id, then indexed by topic_id
    however, beta_normalize_factor and update_beta will be updated during this function, take note that the beta value is not normalized, to normalize the beta, please call normalize_beta
    """
    def update_beta(self, doc, phi_table, beta_normalize_factor, beta):
        # summ up the phi contribution to beta matrix
        for term in doc.keys():
            if len(beta_normalize_factor)==0:
                # if this is the first term ever
                beta_normalize_factor = deepcopy(phi_table[term])
                beta[term] = phi_table[term]
            elif term not in beta.keys():
                # if this is the first time this term appears
                beta[term] = phi_table[term]
                for k in range(self._K):
                    # update the beta normalize factors
                    beta_normalize_factor[k] = log_add(beta_normalize_factor[k], phi_table[term][k])
            else:
                # iterate over all topics
                for k in range(self._K):
                    # update the beta matrix
                    beta[term][k] = log_add(beta[term][k], phi_table[term][k])
                    beta_normalize_factor[k] = log_add(beta_normalize_factor[k], phi_table[term][k])

        return beta, beta_normalize_factor
    
    """
    @param beta: a defaultdict(dict) data type represents the beta matrix, in size of V-by-K, first indexed by term_id, then indexed by topic_id
    @param beta_normalize_factor: a dict data type represents the beta normalize factor
    doc and phi_table will not be modified, however, beta will be updated during this function
    """
    def normalize_beta(self, beta_normalize_factor, update_beta):
        # summ up the phi contribution to beta matrix
        for term in update_beta.keys():
            # iterate over all topics
            for k in range(self._K):
                # update the beta matrix
                update_beta[term][k] -= beta_normalize_factor[k]
                
        return update_beta

    """
    @param alpha_vector: a dict data type represents dirichlet prior, indexed by topic_id
    @param alpha_sufficient_statistics: a dict data type represents alpha sufficient statistics for alpha updating, indexed by topic_id
    alpha_sufficient_statistics value will not be modified, however, alpha_vector will be updated during this function
    """
    def update_alpha(self, alpha_vector, alpha_sufficient_statistics):
        alpha_vector_update = {}
        alpha_gradient = {}
        alpha_hessian = {}
        
        alpha_sum = 0.0
        for k in range(self._K):
            alpha_vector[k] = alpha_vector[k]
            alpha_sum += alpha_vector[k]

        decay = 0
        
        for alpha_iteration in range(self._alpha_maximum_iteration):
            sum_g_h = 0.0
            sum_1_h = 0.0
            
            for k in range(self._K):
                # compute the alpha gradient
                alpha_gradient[k] = self._D * (psi(alpha_sum) - psi(alpha_vector[k])) + alpha_sufficient_statistics[k]
                # compute the alpha hessian
                alpha_hessian[k] = -self._D * polygamma(1, alpha_vector[k])
                
                # if the alpha gradient is not well defined
                if isinf(alpha_gradient[k]) or isnan(alpha_gradient[k]):
                    print "illegal alpha gradient value at index ", k, ": ", alpha_gradient[k]
                
                sum_g_h += alpha_gradient[k] / alpha_hessian[k]
                sum_1_h += 1.0 / alpha_hessian[k]

            z = self._D * polygamma(1, alpha_sum)
            c = sum_g_h / (1.0 / z + sum_1_h)

            # update the alpha vector
            while True:
                singular_hessian = False

                for k in range(self._K):
                    # compute the step size
                    step_size = pow(self._alpha_update_decay_factor, decay) * (alpha_gradient[k] - c) / alpha_hessian[k]
                    if alpha_vector[k] <= step_size:
                        singular_hessian = True
                        break
                    
                    # commit the change to alpha vector
                    alpha_vector_update[k] = alpha_vector[k] - step_size

                if singular_hessian:
                    # if the hessian matrix is a singular matrix, increase the decay power, i.e., reduce the step size
                    decay += 1
                    # revoke the commit
                    alpha_vector_update = alpha_vector
                    if decay > self._alpha_maximum_decay:
                        break
                else:
                    #print "update alpha for decay factor ", pow(self._alpha_update_decay_factor, decay)
                    break
                
            # compute the alpha sum
            # check the alpha converge criteria
            alpha_sum = 0.0
            keep_going = False
            for k in range(self._K):
                alpha_sum += alpha_vector_update[k]
                if abs((alpha_vector_update[k] - alpha_vector[k]) / alpha_vector[k]) >= self._alpha_converge:
                    keep_going = True
            
            # update the alpha vector
            alpha_vector = alpha_vector_update
            
            # break the updating if alpha is converged
            if not keep_going:
                break

        return alpha_vector

    def learning(self):
        old_likelihood = 0.0
        
        for i in range(self._maximum_iteration):
            new_likelihood = self.inference()
            print "em iteration is ", (i+1), " likelihood is ", new_likelihood
            
            if abs((new_likelihood - old_likelihood)/old_likelihood) < self._converge:
                break
            
            old_likelihood = new_likelihood
            
            print "alpha vector is ", self._alpha
            
        print "learning finished..."
            
if __name__ == "__main__":
    from io.de_news_io import parse_de_news_vi
    d = parse_de_news_vi("../../data/de-news/*.en.txt", 'english', 1, 0.4, 0.0001)
    
    print d
    
    lda = VariationalInference();
    lda._initialize(d, 3);
    lda.learning();