"""
Author: Ke Zhai (zhaike@cs.umd.edu)

Implements collapsed Gibbs sampling for the hierarchical Dirichlet process (HDP).
"""

import numpy, scipy;
import scipy.special;

# We will be taking log(0) = -Inf, so turn off this warning
numpy.seterr(divide='ignore')

class CollapsedGibbsSampling(object):
    import scipy.stats;
    
    """
    @param truncation_level: the maximum number of clusters, used for speeding up the computation
    @param snapshot_interval: the interval for exporting a snapshot of the model
    """
    def __init__(self,
                 truncation_level=100,
                 snapshot_interval = 100):
        self._truncation_level = truncation_level;
        self._snapshot_interval = snapshot_interval;

        self._label_title = "Label-";
        self._mu_title = "Mu-";
        self._sigma_title = "Sigma-";
        self._hyper_parameter_vector_title = "Hyper-parameter-vector-";
        self._hyper_parameter_matrix_title = "Hyper-parameter-matrix-";

    """
    @param data: a N-by-D numpy array object, defines N points of D dimension
    @param alpha: the concentration parameter of the dirichlet process
    @param kappa_0: initial kappa_0
    @param nu_0: initial nu_0
    @param mu_0: initial cluster center
    @param lambda_0: initial lambda_0
    """
    def _initialize(self, data, K=1, alpha=1., gamma=1., eta=1.):
        # initialize the total number of topics.
        self._K = K;
        
        # initialize alpha
        self._alpha = alpha;
        # initialize gamma
        self._gamma = gamma;
        # initialize eta
        self._eta = eta;

        # initialize the documents, key by the document path, value by a list of non-stop and tokenized words, with duplication.
        self._corpus = data
        # initialize the size of the collection, i.e., total number of documents.
        self._D = len(self._corpus)
        print self._D, self._corpus

        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._vocab = []
        for token_list in data.values():
            self._total_words = len(token_list);
            self._vocab += token_list;
        self._vocab = list(set(self._vocab));
        
        # initialize the size of the vocabulary, i.e. total number of distinct tokens.
        self._V = len(self._vocab);
        print self._V, self._vocab
        
        # initialize the word count matrix indexed by topic id and word id, i.e., n_{\cdot \cdot k}^v
        self._n_kv = numpy.zeros((self._K, self._V));
        # initialize the word count matrix indexed by topic id and document id, i.e., n_{j \cdot k}
        self._n_dk = numpy.zeros((self._K, self._D));
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
            assert(len(self._k_dt[d])==len(numpy.unique(self._t_dv[d])));
            
            # word_count_table records down the number of words sit on every table
            self._n_dt[d] = numpy.zeros(1) + len(self._corpus[d]);
            assert(len(self._n_dt[d])==len(numpy.unique(self._t_dv[d])));
            #assert(len(self._n_dt[d])==self._T);
            assert(numpy.sum(self._n_dt[d])==len(self._corpus[d]));
            
            for v in self._corpus[d]:
                self._n_kv[0, v] += 1;
            self._n_dk[0, d] = len(self._corpus[d])
            
            self._m_k[0] += len(self._k_dt[d]);
            
        print self._n_kv, self._n_dk, self._m_k
        print self._t_dv, self._k_dt, self._n_dt

    """
    sample the data to train the parameters
    @param iteration: the number of gibbs sampling iteration
    @param directory: the directory to save output, default to "../../output/tmp-output"  
    """
    def sample(self, iteration, directory="../../output/tmp-output/"):
        from nltk.probability import FreqDist;
        
        #sample the total data
        for iter in xrange(iteration):
            print "random sequence of D is", numpy.random.permutation(xrange(self._D))
            for document_index in numpy.random.permutation(xrange(self._D)):
                print "random sequence of document", document_index, "is", numpy.random.permutation(xrange(len(self._corpus[document_index])))
                # sample word assignment
                for word_index in numpy.random.permutation(xrange(len(self._corpus[document_index]))):
                    self.update_params(document_index, word_index, -1);
                    
                    word_id = self._corpus[document_index][word_index];

                    # compute p()
                    f = numpy.zeros(self._K);
                    f_new = self._gamma / self._V;
                    for k in xrange(self._K):
                        f[k] = (self._n_kv[k, word_id] + self._eta)/(numpy.sum(self._n_kv, axis=0)[k] + self._V * self._eta);
                        f_new += self._m_k[k] * f[k];
                    f_new /= (numpy.sum(self._m_k) + self._gamma);
                    
                    p_table = numpy.zeros(len(self._k_dt[document_index])+1);
                    for t in xrange(len(self._k_dt[document_index])):
                        if self._n_dt[document_index][t] > 0:
                            assigned_topic = self._k_dt[document_index][t];
                            assert(assigned_topic>=0 or assigned_topic<self._K);
                            p_table[t] = f[assigned_topic] * self._n_dt[document_index][t];
                        else:
                            p_table[t] = 0.;
                    # compute the probability of assign a word to new table
                    p_table[len(self._k_dt[document_index])] = self._alpha * f_new;

                    cdf = numpy.cumsum(p_table);
                    t_new = numpy.uint8(numpy.nonzero(cdf>=numpy.random.random())[0][0]);
                    print "t_new is", t_new

                    # assign current word to new table
                    self._t_dv[document_index][word_index] = t_new
                    
                    p_topic = numpy.zeros(self._K+1);
                    # if current word sits on a new table, we need to get the topic of that table
                    if t_new==len(self._k_dt[document_index]):
                        # expand the vectors to fit in new table
                        self._n_dt = numpy.hstack((self._n_dt, numpy.zeros(1)));
                        self._k_dt[document_index] = numpy.hstack((self._t_dv[document_index], numpy.zeros(1)));

                        for k in xrange(self._K):
                            p_topic[k] = self._m_k[k] * f[k];
                        p_topic[self._K] = self._gamma/self._V;
                        
                        cdf = numpy.cumsum(p_topic);
                        k_new = numpy.uint8(numpy.nonzero(cdf>=numpy.random.random())[0][0]);
                        
                        # if current table requires a new topic
                        if k_new==self._K:
                            # expand the matrices to fit in new topic
                            self._K += 1;
                            self._n_kv = numpy.vstack((self._n_kv, numpy.zeros((1, self._V))));
                            assert(self._n_kv.shape==(self._K, self._V));

                            self._n_dk = numpy.vstack((self._n_dk, numpy.zeros((1, self._D))));
                            assert(self._n_dk.shape==(self._K, self._D));
                            
                            self._m_k = numpy.hstack((self._m_k, numpy.zeros(1)));
                            assert(len(self._m_k)==self._K);
                    
                        self.update_params(document_index, word_index, +1);
                    else:
                        self.update_params(document_index, word_index, +1);
                        
                # sample table assignment
                for table_index in numpy.random.permutation(xrange(len(self._k_dt[document_index]))):
                    # if this table is not empty, sample the topic assignment of this table
                    if self._n_dt[document_index][table_index]>0:
                        
                        q = numpy.zeros(self._K+1);

                        # find the index of the words sitting on the current table
                        selected_word_index = numpy.nonzero(self._t_dv[document_index]==table_index)[0];
                        # find the frequency distribution of the words sitting on the current table
                        selected_word_freq_dist = FreqDist(list(self._corpus[selected_word_index]));
                        
                        q[self._K] = scipy.special.gammaln(self._V * self._eta) - scipy.special.gammaln(self._n_dt[document_index][table_index] + self._V * self._eta);
                        for word_id in selected_word_freq_dist.keys():
                            q[self._K] += scipy.special.gammaln(selected_word_freq_dist[word_id] + self._eta) - scipy.special.gammaln(self._eta);
                        q[self._K] += numpy.log(self._gamma);
                        
                        
                        
                        
    
"""

        # initialize the word count matrix indexed by topic id and word id, i.e., n_{\cdot \cdot k}^v
        self._n_kv = numpy.zeros((self._K, self._V));
        # initialize the word count matrix indexed by topic id and document id, i.e., n_{j \cdot k}
        self._n_dk = numpy.zeros((self._K, self._D));
        # initialize the table count matrix indexed by topic id, i.e., m_{\cdot k}
        self._m_k = numpy.zeros(self._K);

        # initialize the table information vectors indexed by document id and word id, i.e., t{j i}
        self._t_dv = {};
        # initialize the topic information vectors indexed by document id and table id, i.e., k_{j t}
        self._k_dt = {};
        # initialize the word count vectors indexed by document id and table id, i.e., n_{j t \cdot}
        self._n_dt = {};
"""    
    
    
    """
    """
    def update_params(self, document_index, word_index, update):
        table_id = self._t_dv[document_index][word_index];
        topic_id = self._k_dt[document_index][table_id];
        assert(topic_id>=0);
        word_id = self._corpus[document_index][word_index];

        self._n_dt[document_index][table_id] += update;
        self._n_kv[topic_id, word_id] += update;
        
        self._n_dk[topic_id, document_index] += update;
        
        # if current table in current document becomes empty 
        if update==-1 and self._n_dt[document_index][table_id]==0:
            # adjust the table counts
            self._m_k[topic_id] -= 1;
            # clear the topic assign of that table
            self._k_dt[document_index][table_id] = -1;
            
        # if a new table is created in current document
        if update==1 and self._n_dt[document_index][table_id]==1:
            # adjust the table counts
            self._m_k[topic_id] += 1;
            # clear the topic assign of that table
            self._k_dt[document_index][table_id] = +1;

    """
    """
    def export_snapshot(self, directory, index):
        import os
        if not os.path.exists(directory):
            os.mkdir(directory);
        assert(directory.endswith("/"));
        
        numpy.savetxt(directory + self._label_title + str(index), numpy.uint8(self._label));
        numpy.savetxt(directory + self._mu_title + str(index), self._sum[:self._K, :]/self._count[:self._K][numpy.newaxis, :].transpose());
        sigma = self._sigma_inv;
        for k in xrange(self._K):
            sigma[k, :, :] = numpy.linalg.inv(sigma[k, :, :]);
        numpy.savetxt(directory + self._sigma_title + str(index), numpy.reshape(sigma[:self._K, :, :], (self._K, self._D*self._D)));
        vector = numpy.array([self._alpha, self._kappa_0, self._nu_0]);
        numpy.savetxt(directory + self._hyper_parameter_vector_title + str(index), vector);
        matrix = numpy.vstack((self._mu_0, self._lambda_0));
        numpy.savetxt(directory + self._hyper_parameter_matrix_title, matrix);
        
        print "successfully export the snapshot to " + directory + " for iteration " + str(index) + "..."
        
"""
"""
def import_monolingual_data(input_file):
    import codecs
    input = codecs.open(input_file, mode="r", encoding="utf-8")
    
    doc_count = 0
    docs = {}
    
    for line in input:
        line = line.strip().lower();

        contents = line.split("\t");
        assert(len(contents)==2);
        docs[int(contents[0])] = [int(item) for item in contents[1].split()];

        doc_count+=1
        if doc_count%10000==0:
            print "successfully import " + str(doc_count) + " documents..."

    print "successfully import all documents..."
    return docs

"""
run IGMM on the synthetic clustering dataset.
"""
if __name__ == '__main__':
    temp_directory = "../../data/test/";
    data = import_monolingual_data(temp_directory + "doc.dat");

    gs = CollapsedGibbsSampling();
    gs._initialize(data);
    
    gs.sample(10);
    
    print gs._K
