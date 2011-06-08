"""
@author: Jordan Boyd-Graber (jbg@umiacs.umd.edu)
@author: Ke Zhai (zhaike@cs.umd.edu)
"""

import math, random;
import scipy;
import util.log_math;

from collections import defaultdict
from nltk import FreqDist

"""
This is a python implementation of lda, based on collapsed Gibbs sampling, with hyper parameter updating.
It only supports symmetric Dirichlet prior over the topic simplex.

References:
[1] T. L. Griffiths & M. Steyvers. Finding Scientific Topics. Proceedings of the National Academy of Sciences, 101, 5228-5235, 2004.
"""
class CollapsedGibbsSampling:
    """
    
    """
    def __init__(self, alpha=0.5, beta=0.1, 
                 gibbs_sampling_maximum_iteration=200, 
                 hyper_parameter_maximum_iteration=100, 
                 hyper_parameter_sampling_interval=25):
        # set the document smooth factor
        self._alpha = alpha
        # set the vocabulary smooth factor
        self._beta = beta
        
        self._hyper_parameter_maximum_iteration = hyper_parameter_maximum_iteration
        self._gibbs_sampling_maximum_iteration = gibbs_sampling_maximum_iteration
        self._hyper_parameter_sampling_interval = hyper_parameter_sampling_interval;
        assert(self._hyper_parameter_sampling_interval>0);
        
        # pending for further changing~
        #self._hyper_parameter_converge_threshold = 0.000001
        #self._variational_inference_converge_threshold = 0.00001
        #self._gamma_converge_threshold = 0.000001
        #self._gamma_maximum_iteration = 400
        
    """
    @param num_topics: desired number of topics
    @param data: a dict data type, indexed by document id, value is a list of words in that document, not necessarily be unique
    """
    def _initialize(self, data, num_topics=10):
        # define the counts over different topics for all documents, first indexed by doc id, the indexed by topic id
        self._doc_topics = defaultdict(FreqDist)
        # define the counts over words for all topics, first indexed by topic id, then indexed by token id
        self._topic_words = defaultdict(FreqDist)
        # define the topic assignment for every word in every document, first indexed by doc id, then indexed by word position
        self._topic_assignment = defaultdict(dict)
        
        self._K = num_topics
    
        self._alpha_sum = self._alpha * self._K

        # define the input data
        self._data = data
        # define the total number of document
        self._D = len(data)
    
        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._vocab = set([])
        for doc in self._data:
            for position in xrange(len(self._data[doc])):
                # learn all the words we'll see
                self._vocab.add(self._data[doc][position])
            
                # initialize the state to unassigned
                self._topic_assignment[doc][position] = -1
                
        self._V = len(self._vocab)
        
    """
    
    """
    def optimize_hyperparameters(self, samples=5, step=3.0):
        rawParam = [math.log(self._alpha), math.log(self._beta)]

        for ii in xrange(samples):
            log_likelihood_old = self.compute_likelihood(self._alpha, self._beta)
            log_likelihood_new = math.log(random.random()) + log_likelihood_old
            print("OLD: %f\tNEW: %f at (%f, %f)" % (log_likelihood_old, log_likelihood_new, self._alpha, self._beta))

            l = [x - random.random() * step for x in rawParam]
            r = [x + step for x in rawParam]

            for jj in xrange(self._hyper_parameter_maximum_iteration):
                rawParamNew = [l[x] + random.random() * (r[x] - l[x]) for x in xrange(len(rawParam))]
                trial_alpha, trial_beta = [math.exp(x) for x in rawParamNew]
                lp_test = self.compute_likelihood(trial_alpha, trial_beta)

                if lp_test > log_likelihood_new:
                    print(jj)
                    self._alpha = math.exp(rawParamNew[0])
                    self._beta = math.exp(rawParamNew[1])
                    self._alpha_sum = self._alpha * self._K
                    self._beta_sum = self._beta * self._V
                    rawParam = [math.log(self._alpha), math.log(self._beta)]
                    break
                else:
                    for dd in xrange(len(rawParamNew)):
                        if rawParamNew[dd] < rawParam[dd]:
                            l[dd] = rawParamNew[dd]
                        else:
                            r[dd] = rawParamNew[dd]
                        assert l[dd] <= rawParam[dd]
                        assert r[dd] >= rawParam[dd]

            print("\nNew hyperparameters (%i): %f %f" % (jj, self._alpha, self._beta))

    """
    compute the log-likelihood of the model
    """
    def compute_likelihood(self, alpha, beta):
        assert len(self._doc_topics) == self._D
        
        alpha_sum = alpha * self._K
        beta_sum = beta * self._V

        likelihood = 0.0
        # compute the log likelihood of the document
        likelihood += scipy.special.gammaln(alpha_sum) * len(self._data)
        likelihood -= scipy.special.gammaln(alpha) * self._K * len(self._data)
           
        for ii in self._doc_topics.keys():
            for jj in xrange(self._K):
                likelihood += scipy.special.gammaln(alpha + self._doc_topics[ii][jj])                    
            likelihood -= scipy.special.gammaln(alpha_sum + self._doc_topics[ii].N())
            
        # compute the log likelihood of the topic
        likelihood += scipy.special.gammaln(beta_sum) * self._K
        likelihood -= scipy.special.gammaln(beta) * self._V * self._K
            
        for ii in self._topic_words.keys():
            for jj in self._vocab:
                likelihood += scipy.special.gammaln(beta + self._topic_words[ii][jj])
            likelihood -= scipy.special.gammaln(beta_sum + self._topic_words[ii].N())
            
        return likelihood

    """
    compute the conditional distribution
    @param doc: doc id
    @param word: word id
    @param topic: topic id  
    @return: the probability value of the topic for that word in that document
    """
    def log_prob(self, doc, word, topic):
        val = math.log(self._doc_topics[doc][topic] + self._alpha)
        #this is constant across a document, so we don't need to compute this term
        # val -= math.log(self._doc_topics[doc].N() + self._alpha_sum)
        
        val += math.log(self._topic_words[topic][word] + self._beta)
        val -= math.log(self._topic_words[topic].N() + self._V * self._beta)
    
        return val

    """
    this method samples the word at position in document, by covering that word and compute its new topic distribution, in the end, both self._topic_assignment, self._doc_topics and self._topic_words will change
    @param doc: a document id
    @param position: the position in doc, ranged as range(self._data[doc])
    """
    def sample_word(self, doc, position):        
        assert position >= 0 and position < len(self._data[doc])
        
        #retrieve the word
        word = self._data[doc][position]
    
        #get the old topic assignment to the word in doc at position
        old_topic = self._topic_assignment[doc][position]
        if old_topic != -1:
            #this word already has a valid topic assignment, decrease the topic|doc counts and word|topic counts by covering up that word
            self.change_count(doc, word, old_topic, -1)

        #compute the topic probability of current word, given the topic assignment for other words
        probs = [self.log_prob(doc, self._data[doc][position], x) for x in xrange(self._K)]

        #sample a new topic out of a distribution according to probs
        new_topic = util.log_math.log_sample(probs)

        #after we draw a new topic for that word, we will change the topic|doc counts and word|topic counts, i.e., add the counts back
        self.change_count(doc, word, new_topic, 1)
        #assign the topic for the word of current document at current position
        self._topic_assignment[doc][position] = new_topic

    """
    this methods change the count of a topic in one doc and a word of one topic by delta
    this values will be used in the computation
    @param doc: the doc id
    @param word: the word id
    @param topic: the topic id
    @param delta: the change in the value
    """
    def change_count(self, doc, word, topic, delta):
        self._doc_topics[doc].inc(topic, delta)
        self._topic_words[topic].inc(word, delta)

    """
    sample the corpus to train the parameters
    @param hyper_delay: defines the delay in updating they hyper parameters, i.e., start updating hyper parameter only after hyper_delay number of gibbs sampling iterations. Usually, it specifies a burn-in period.
    """
    def sample(self):
        assert self._topic_assignment
        
        #sample the total corpus
        for iter in xrange(self._gibbs_sampling_maximum_iteration):
            #sample every document
            for doc in self._data:
                #sample every position
                for position in xrange(len(self._data[doc])):
                    self.sample_word(doc, position)
                    
            print("iteration %i %f" % (iter, self.compute_likelihood(self._alpha, self._beta)))
            if iter % self._hyper_parameter_sampling_interval == 0:
                self.optimize_hyperparameters()

    def print_topics(self, num_words=15):
        for ii in self._topic_words:
            print("%i:%s\n" % (ii, "\t".join(self._topic_words[ii].keys()[:num_words])))

if __name__ == "__main__":
    from io.de_news_io import parse_to_gs_format
    d = parse_to_gs_format("../../data/de-news/*.en.txt", "english", 100, 0.3, 0.0001)
    
    lda = CollapsedGibbsSampling()
    lda._initialize(d, 3)

    lda.sample(25)
    lda.print_topics()