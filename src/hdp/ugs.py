"""
@author: Ke Zhai (zhaike@cs.umd.edu)

This code was modified from the code originally written by Chong Wang (chongw@cs.princeton.edu).
Implements uncollapsed Gibbs sampling for the hierarchical Dirichlet process (HDP).

References:
[1] Chong Wang and David Blei, A Split-Merge MCMC Algorithm for the Hierarchical Dirichlet Process, available online www.cs.princeton.edu/~chongw/papers/sm-hdp.pdf.
"""

import numpy, scipy;
import scipy.special;

# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')

class UncollapsedGibbsSampling(object):
    import scipy.stats;
    
    """
    @param truncation_level: the maximum number of clusters, used for speeding up the computation
    @param snapshot_interval: the interval for exporting a snapshot of the model
    """
    def __init__(self,
                 #truncation_level=100,
                 snapshot_interval=100):
        #self._truncation_level = truncation_level;
        self._snapshot_interval = snapshot_interval;

        self._table_info_title = "Table-information-";
        self._topic_info_title = "Topic-information-";
        self._hyper_parameter_title = "Hyper-parameter-";

    """
    @param data: a N-by-D numpy array object, defines N points of D dimension
    @param K: number of topics, number of broke sticks
    @param alpha: the probability of f_{k_{\mathsf{new}}}^{-x_{dv}}(x_{dv}), the prior probability density for x_{dv}
    @param gamma: the smoothing value for a table to be assigned to a new topic
    @param eta: the smoothing value for a word to be assigned to a new topic
    """
    def _initialize(self, data, K=1, alpha=1., gamma=1., eta=1.):
        # initialize the total number of topics.
        self._K = K;
        
        # initialize alpha, the probability of f_{k_{\mathsf{new}}}^{-x_{dv}}(x_{dv}), the prior probability density for x_{dv}
        self._alpha = alpha;
        # initialize eta, the smoothing value for a word to be assigned to a new topic
        self._eta = eta;
        # initialize gamma, the smoothing value for a table to be assigned to a new topic
        self._gamma = gamma;

        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        self._corpus = data
        # initialize the size of the collection, i.e., total number of documents.
        self._D = len(self._corpus)

        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._vocab = []
        for token_list in data.values():
            self._total_words = len(token_list);
            self._vocab += token_list;
        self._vocab = list(set(self._vocab));
        
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        self._V = len(self._vocab);
        
        # initialize the word count matrix indexed by topic id and word id, i.e., n_{\cdot \cdot k}^v
        self._n_kv = numpy.zeros((self._K, self._V));
        # initialize the word count matrix indexed by topic id and document id, i.e., n_{j \cdot k}
        self._n_kd = numpy.zeros((self._K, self._D));
        # initialize the table count matrix indexed by topic id, i.e., m_{\cdot k}
        self._m_k = numpy.zeros(self._K);

        # initialize the table information vectors indexed by document id and word id, i.e., t{j i}
        self._t_dv = {};
        # initialize the topic information vectors indexed by document id and table id, i.e., k_{j t}
        self._k_dt = {};
        # initialize the word count vectors indexed by document id and table id, i.e., n_{j t \cdot}
        self._n_dt = {};
        
        # we assume all words in a document belong to one table which was assigned to topic 0 
        for d in xrange(self._D):
            # initialize the table information vector indexed by document and records down which table a word belongs to 
            self._t_dv[d] = numpy.zeros(len(self._corpus[d]), dtype=numpy.int);
            
            # self._k_dt records down which topic a table was assigned to
            self._k_dt[d] = numpy.zeros(1, dtype=numpy.int);
            assert(len(self._k_dt[d]) == len(numpy.unique(self._t_dv[d])));
            
            # word_count_table records down the number of words sit on every table
            self._n_dt[d] = numpy.zeros(1, dtype=numpy.int) + len(self._corpus[d]);
            assert(len(self._n_dt[d]) == len(numpy.unique(self._t_dv[d])));
            assert(numpy.sum(self._n_dt[d]) == len(self._corpus[d]));
            
            for v in self._corpus[d]:
                self._n_kv[0, v] += 1;
            self._n_kd[0, d] = len(self._corpus[d])
            
            self._m_k[0] += len(self._k_dt[d]);

    """
    sample the data to train the parameters
    @param iteration: the number of gibbs sampling iteration
    @param directory: the directory to save output, default to "../../output/tmp-output"  
    """
    def sample(self, iteration, directory="../../output/tmp-output/"):
        from nltk.probability import FreqDist;
        
        #sample the total data
        for iter in xrange(iteration):
            for document_index in numpy.random.permutation(xrange(self._D)):
                # sample word assignment, see which table it should belong to
                for word_index in numpy.random.permutation(xrange(len(self._corpus[document_index]))):
                    self.update_params(document_index, word_index, -1);
                    
                    # get the word at the index position
                    word_id = self._corpus[document_index][word_index];

                    n_k = numpy.sum(self._n_kv, axis=1);
                    assert(len(n_k) == self._K);
                    f = numpy.zeros(self._K);
                    f_new = self._gamma / self._V;
                    for k in xrange(self._K):
                        f[k] = (self._n_kv[k, word_id] + self._eta) / (n_k[k] + self._V * self._eta);
                        f_new += self._m_k[k] * f[k];
                    f_new /= (numpy.sum(self._m_k) + self._gamma);
                    
                    # compute the probability of this word sitting at every table 
                    table_probability = numpy.zeros(len(self._k_dt[document_index]) + 1);
                    for t in xrange(len(self._k_dt[document_index])):
                        if self._n_dt[document_index][t] > 0:
                            # if there are some words sitting on this table, the probability will be proportional to the population
                            assigned_topic = self._k_dt[document_index][t];
                            assert(assigned_topic >= 0 or assigned_topic < self._K);
                            table_probability[t] = f[assigned_topic] * self._n_dt[document_index][t];
                        else:
                            # if there are no words sitting on this table
                            # note that it is an old table, hence the prior probability is 0, not self._alpha
                            table_probability[t] = 0.;
                    # compute the probability of current word sitting on a new table, the prior probability is self._alpha
                    table_probability[len(self._k_dt[document_index])] = self._alpha * f_new;

                    # sample a new table this word should sit in
                    table_probability /= numpy.sum(table_probability);
                    cdf = numpy.cumsum(table_probability);
                    new_table = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);

                    # assign current word to new table
                    self._t_dv[document_index][word_index] = new_table;

                    # if current word sits on a new table, we need to get the topic of that table
                    if new_table == len(self._k_dt[document_index]):
                        # expand the vectors to fit in new table
                        self._n_dt[document_index] = numpy.hstack((self._n_dt[document_index], numpy.zeros(1)));
                        self._k_dt[document_index] = numpy.hstack((self._k_dt[document_index], numpy.zeros(1)));
                        
                        assert(len(self._n_dt) == self._D and numpy.all(self._n_dt[document_index] >= 0));
                        assert(len(self._k_dt) == self._D and numpy.all(self._k_dt[document_index] >= 0));
                        assert(len(self._n_dt[document_index]) == len(self._k_dt[document_index]));

                        # compute the probability of this table having every topic
                        topic_probability = numpy.zeros(self._K + 1);
                        for k in xrange(self._K):
                            topic_probability[k] = self._m_k[k] * f[k];
                        topic_probability[self._K] = self._gamma / self._V;

                        # sample a new topic this table should be assigned
                        topic_probability /= numpy.sum(topic_probability);
                        cdf = numpy.cumsum(topic_probability);
                        new_topic = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
                        
                        self._k_dt[document_index][new_table] = new_topic
                        
                        # if current table requires a new topic
                        if new_topic == self._K:
                            # expand the matrices to fit in new topic
                            self._K += 1;
                            self._n_kv = numpy.vstack((self._n_kv, numpy.zeros((1, self._V))));
                            assert(self._n_kv.shape == (self._K, self._V));
                            self._n_kd = numpy.vstack((self._n_kd, numpy.zeros((1, self._D))));
                            assert(self._n_kd.shape == (self._K, self._D));
                            self._k_dt[document_index][-1] = new_topic;
                            self._m_k = numpy.hstack((self._m_k, numpy.zeros(1)));
                            assert(len(self._m_k) == self._K);
                            
                    self.update_params(document_index, word_index, +1);
                        
                # sample table assignment, see which topic it should belong to
                for table_index in numpy.random.permutation(xrange(len(self._k_dt[document_index]))):
                    # if this table is not empty, sample the topic assignment of this table
                    if self._n_dt[document_index][table_index] > 0:
                        old_topic = self._k_dt[document_index][table_index];

                        # find the index of the words sitting on the current table
                        selected_word_index = numpy.nonzero(self._t_dv[document_index] == table_index)[0];
                        # find the frequency distribution of the words sitting on the current table
                        selected_word_freq_dist = FreqDist([self._corpus[document_index][term] for term in list(selected_word_index)]);

                        # compute the probability of assigning current table every topic
                        topic_probability = numpy.zeros(self._K + 1);
                        topic_probability[self._K] = scipy.special.gammaln(self._V * self._eta) - scipy.special.gammaln(self._n_dt[document_index][table_index] + self._V * self._eta);
                        for word_id in selected_word_freq_dist.keys():
                            topic_probability[self._K] += scipy.special.gammaln(selected_word_freq_dist[word_id] + self._eta) - scipy.special.gammaln(self._eta);
                        topic_probability[self._K] += numpy.log(self._gamma);
                        
                        n_k = numpy.sum(self._n_kv, axis=1);
                        assert(len(n_k) == (self._K))
                        for topic_index in xrange(self._K):
                            if topic_index == old_topic:
                                if self._m_k[topic_index] <= 1:
                                    # if current table is the only table assigned to current topic,
                                    # it means this topic is probably less useful or less generalizable to other documents,
                                    # it makes more sense to collapse this topic and hence assign this table to other topic.
                                    topic_probability[topic_index] = -1e500;
                                else:
                                    # if there are other tables assigned to current topic
                                    topic_probability[topic_index] = scipy.special.gammaln(self._V * self._eta + n_k[topic_index] - self._n_dt[document_index][table_index]) - scipy.special.gammaln(self._V * self._eta + n_k[topic_index]);
                                    for word_id in selected_word_freq_dist.keys():
                                        topic_probability[topic_index] += scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._eta) - scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._eta - selected_word_freq_dist[word_id]);
                                    # compute the prior if we move this table from this topic
                                    topic_probability[topic_index] += numpy.log(self._m_k[topic_index] - 1);
                            else:
                                topic_probability[topic_index] = scipy.special.gammaln(self._V * self._eta + n_k[topic_index]) - scipy.special.gammaln(self._V * self._eta + n_k[topic_index] + self._n_dt[document_index][table_index]);
                                for word_id in selected_word_freq_dist.keys():
                                    topic_probability[topic_index] += scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._eta + selected_word_freq_dist[word_id]) - scipy.special.gammaln(self._n_kv[topic_index, word_id] + self._eta);
                                topic_probability[topic_index] += numpy.log(self._m_k[topic_index]);

                        # normalize the distribution and sample new topic assignment for this topic
                        #topic_probability = numpy.exp(topic_probability);
                        #topic_probability = topic_probability/numpy.sum(topic_probability);
                        topic_probability = numpy.exp(log_normalize(topic_probability));
                        cdf = numpy.cumsum(topic_probability);
                        new_topic = numpy.uint8(numpy.nonzero(cdf >= numpy.random.random())[0][0]);
                        
                        # if the table is assigned to a new topic
                        if new_topic != old_topic:
                            # assign this table to new topic
                            self._k_dt[document_index][table_index] = new_topic;
                            
                            # if this table starts a new topic, expand all matrix
                            if new_topic == self._K:
                                self._K += 1;
                                self._n_kd = numpy.vstack((self._n_kd, numpy.zeros((1, self._D))));
                                assert(self._n_kd.shape == (self._K, self._D));
                                self._n_kv = numpy.vstack((self._n_kv, numpy.zeros((1, self._V))));
                                assert(self._n_kv.shape == (self._K, self._V));
                                self._m_k = numpy.hstack((self._m_k, numpy.zeros(1)));
                                assert(len(self._m_k) == self._K);
                                
                            # adjust the statistics of all model parameter
                            self._m_k[old_topic] -= 1;
                            self._m_k[new_topic] += 1;
                            self._n_kd[old_topic, document_index] -= self._n_dt[document_index][table_index];
                            self._n_kd[new_topic, document_index] += self._n_dt[document_index][table_index];
                            for word_id in selected_word_freq_dist.keys():
                                self._n_kv[old_topic, word_id] -= selected_word_freq_dist[word_id];
                                assert(self._n_kv[old_topic, word_id] >= 0)
                                self._n_kv[new_topic, word_id] += selected_word_freq_dist[word_id];

            # compact all the parameters, including removing unused topics and unused tables
            self.compact_params();
            
            if iter > 0 and iter % 10 == 0:
                print "sampling in progress %2d%%" % (100 * iter / iteration);
                print "total number of topics %i, log-likelihood is %f" % (self._K, self.log_likelihood());
                
            if (iter + 1) % self._snapshot_interval == 0:
                self.export_snapshot(directory, iter + 1);
                
    """
    @param document_index: the document index to update
    @param word_index: the word index to update
    @param update: the update amount for this document and this word
    @attention: the update table index and topic index is retrieved from self._t_dv and self._k_dt, so make sure these values were set properly before invoking this function
    """
    def update_params(self, document_index, word_index, update):
        # retrieve the table_id of the current word of current document
        table_id = self._t_dv[document_index][word_index];
        # retrieve the topic_id of the table that current word of current document sit on
        topic_id = self._k_dt[document_index][table_id];
        # get the word_id of at the word_index of the document_index
        word_id = self._corpus[document_index][word_index];

        self._n_dt[document_index][table_id] += update;
        assert(numpy.all(self._n_dt[document_index] >= 0));
        self._n_kv[topic_id, word_id] += update;
        assert(numpy.all(self._n_kv >= 0));
        self._n_kd[topic_id, document_index] += update;
        assert(numpy.all(self._n_kd >= 0));
        
        # if current table in current document becomes empty 
        if update == -1 and self._n_dt[document_index][table_id] == 0:
            # adjust the table counts
            self._m_k[topic_id] -= 1;
            
        # if a new table is created in current document
        if update == 1 and self._n_dt[document_index][table_id] == 1:
            # adjust the table counts
            self._m_k[topic_id] += 1;
            
        assert(numpy.all(self._m_k >= 0));
        assert(numpy.all(self._k_dt[document_index] >= 0));

    """
    """
    def compact_params(self):
        # find unused and used topics
        unused_topics = numpy.nonzero(self._m_k == 0)[0];
        used_topics = numpy.nonzero(self._m_k != 0)[0];
        
        self._K -= len(unused_topics);
        assert(self._K >= 1 and self._K == len(used_topics));
        
        self._n_kd = numpy.delete(self._n_kd, unused_topics, axis=0);
        assert(self._n_kd.shape == (self._K, self._D));
        self._n_kv = numpy.delete(self._n_kv, unused_topics, axis=0);
        assert(self._n_kv.shape == (self._K, self._V));
        self._m_k = numpy.delete(self._m_k, unused_topics);
        assert(len(self._m_k) == self._K);
        
        for d in xrange(self._D):
            # find the unused and used tables
            unused_tables = numpy.nonzero(self._n_dt[d] == 0)[0];
            used_tables = numpy.nonzero(self._n_dt[d] != 0)[0];

            self._n_dt[d] = numpy.delete(self._n_dt[d], unused_tables);
            self._k_dt[d] = numpy.delete(self._k_dt[d], unused_tables);
            
            # shift down all the table indices of all words in current document
            # @attention: shift the used tables in ascending order only.
            for t in xrange(len(self._n_dt[d])):
                self._t_dv[d][numpy.nonzero(self._t_dv[d] == used_tables[t])[0]] = t;
            
            # shrink down all the topics indices of all tables in current document
            # @attention: shrink the used topics in ascending order only.
            for k in xrange(self._K):
                self._k_dt[d][numpy.nonzero(self._k_dt[d] == used_topics[k])[0]] = k;

    """
    compute the log likelihood of the model
    """
    def log_likelihood(self):
        log_likelihood = 0.;
        # compute the document level log likelihood
        log_likelihood += self.table_log_likelihood();
        # compute the table level log likelihood
        log_likelihood += self.topic_log_likelihood();
        # compute the word level log likelihood
        log_likelihood += self.word_log_likelihood();
        
        #todo: add in the likelihood for hyper-parameter
        
        return log_likelihood
        
    """
    compute the table level prior in log scale \prod_{d=1}^D (p(t_{d})), where p(t_d) = \frac{ \alpha^m_d \prod_{t=1}^{m_d}(n_di-1)! }{ \prod_{v=1}^{n_d}(v+\alpha-1) }
    """
    def table_log_likelihood(self):
        log_likelihood = 0.;
        for document_index in xrange(self._D):
            log_likelihood += len(self._k_dt[document_index]) * numpy.log(self._alpha) - log_factorial(len(self._t_dv[document_index]), self._alpha);
            for table_index in xrange(len(self._k_dt[document_index])):
                log_likelihood += scipy.special.gammaln(self._n_dt[document_index][table_index]);
            
        return log_likelihood
    
    """
    compute the topic level prior in log scale p(k) = \frac{ \gamma^K \prod_{k=1}^{K}(m_k-1)! }{ \prod_{s=1}^{m}(s+\gamma-1) }
    """
    def topic_log_likelihood(self):
        log_likelihood = self._K * numpy.log(self._gamma) - log_factorial(numpy.sum(self._m_k), self._gamma);
        for topic_index in xrange(self._K):
            log_likelihood += scipy.special.gammaln(self._m_k[topic_index]);
        
        return log_likelihood
    
    """
    compute the word level log likelihood p(x | t, k) = \prod_{k=1}^K f(x_{ij} | z_{ij}=k), where f(x_{ij} | z_{ij}=k) = \frac{\Gamma(V \eta)}{\Gamma(n_k + V \eta)} \frac{\prod_{v} \Gamma(n_{k}^{v} + \eta)}{\Gamma^V(\eta)}
    """
    def word_log_likelihood(self):
        n_k = numpy.sum(self._n_kd, axis=1);
        assert(len(n_k) == self._K);
        
        log_likelihood = self._K * scipy.special.gammaln(self._V * self._eta);
        for topic_index in xrange(self._K):
            log_likelihood -= scipy.special.gammaln(self._V * self._eta + n_k[topic_index]);
            for word_index in xrange(self._V):
                if self._n_kv[topic_index, word_index] > 0:
                    log_likelihood += scipy.special.gammaln(self._n_kv[topic_index, word_index] + self._eta) + scipy.special.gammaln(self._eta);
                    
        return log_likelihood
        
    """
    """
    def export_snapshot(self, directory, index):
        import os
        if not os.path.exists(directory):
            os.mkdir(directory);
        assert(directory.endswith("/"));
        
        hyper_parameter = numpy.array([self._alpha, self._eta, self._gamma]);
        numpy.savetxt(directory + self._hyper_parameter_title + str(index), hyper_parameter);
        
        output1 = open(directory + self._table_info_title + str(index), 'w');
        output2 = open(directory + self._topic_info_title + str(index), 'w');
        for d in xrange(self._D):
            output1.write(" ".join([str(item) for item in self._t_dv[d]]) + "\n");
            output2.write(" ".join([str(item) for item in self._k_dt[d]]) + "\n");
            
        print "successfully export the snapshot to " + directory + " for iteration " + str(index) + "..."

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

from math import log, exp;

def log_add(log_a, log_b):
    if log_a < log_b:
        return log_b + log(1 + exp(log_a - log_b))
    else:
        return log_a + log(1 + exp(log_b - log_a))

def log_normalize(dist):
    normalizer = reduce(log_add, dist)
    for ii in xrange(len(dist)):
        dist[ii] -= normalizer
    return dist

"""
@param n: an integer data type
@param a: 
@attention: n must be an integer
this function is to compute the log(n!), since n!=Gamma(n+1), which means log(n!)=lngamma(n+1)
"""
def log_factorial(n, a):
    if n == 0:
        return 0.;
    return scipy.special.gammaln(n + a) - scipy.special.gammaln(a);

"""
"""
def print_topics(n_kv, term_mapping, top_words=10):
    input = open(term_mapping);
    vocab = {};
    i = 0;
    for line in input:
        vocab[i] = line.strip();
        i += 1;

    (K, V) = n_kv.shape;
    assert(V == len(vocab));

    if top_words >= V:
        sorted_counts = numpy.zeros((1, K)) - numpy.log(V);
    else:
        sorted_counts = numpy.sort(n_kv, axis=1);
        sorted_counts = sorted_counts[:, -top_words][:, numpy.newaxis];
    
    assert(sorted_counts.shape==(K, 1));

    for k in xrange(K):
        display = (n_kv[[k], :] >= sorted_counts[k, :]);
        assert(display.shape == (1, V));
        output_str = str(k) + ": ";
        for v in xrange(self._V):
            if display[:, v]:
                output_str += vocab[v] + "\t";
        print output_str

"""
run HDP on a synthetic corpus.
"""
if __name__ == '__main__':
    temp_directory = "../../data/test/";
    #temp_directory = "../../data/de-news/en/corpus-3/";
    data = import_monolingual_data(temp_directory + "doc.dat");
    print data

    gs = UncollapsedGibbsSampling(50);
    gs._initialize(data);
    
    gs.sample(200);
    
    print gs._K;
    print gs._n_kv