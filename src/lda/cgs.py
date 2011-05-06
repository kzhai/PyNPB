#!/usr/bin/pyhon

# Original Author: Jordan Boyd-Graber
# Email: jbg@umiacs.umd.edu

# Modification: Ke Zhai
# Email: zhaike@cs.umd.edu

#TODO: buggy version, under modication

from collections import defaultdict
from math import log, exp
from random import random
from nltk import FreqDist
from scipy.special import gamma, psi, gammaln, polygamma
from util.log_math import log_sample

class CollapsedGibbsSampling:
    def __init__(self):
        # set the document smooth factor
        self._alpha = 0.5
        # set the vocabulary smooth factor
        self._beta = 0.1
        
        self._alpha_converge = 0.000001
        self._alpha_maximum_iteration = 100
        
        self._maximum_iteration = 200
        self._converge = 0.00001

        # pending for further changing~
        self._gamma_converge = 0.000001
        self._gamma_maximum_iteration = 400
        
    """
    @param num_topics: desired number of topics
    @param data: a dict data type, indexed by document id, value is a list of words in that document, not necessarily be unique
    """
    def _initialize(self, data, num_topics=10):
        #define the counts over different topics for all documents, first indexed by doc id, the indexed by topic id
        self._doc_topics = defaultdict(FreqDist)
        #define the counts over words for all topics, first indexed by topic id, then indexed by token id
        self._topic_words = defaultdict(FreqDist)
        #define the topic assignment for every word in every document, first indexed by doc id, then indexed by word position
        self._topic_assignment = defaultdict(dict)
        
        self._K = num_topics
        
        #initialize a K-dimensional vector, valued at 1/K
        #self._alpha = []
        self._alpha_sum = self._alpha * self._K
        #for k in xrange(self._K):
        #    self._alpha.append(random() / self._K)
        #    self._alpha_sum = self._alpha_sum + self._alpha[k]
        #print self._alpha
        
        #self._alpha_sum = self._alpha * self._K
    
        self._data = data
        #define the total number of document
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
        
        #self._beta_sum = float(self._V) * self._beta

    """
    
    """
    def optimize_hyperparameters(self, samples=5, step=3.0):
        rawParam = [log(self._alpha), log(self._beta)]

        for ii in xrange(samples):
            lp_old = self.compute_likelihood(self._alpha, self._beta)
            lp_new = log(random()) + lp_old
            print("OLD: %f\tNEW: %f at (%f, %f)" % (lp_old, lp_new, self._alpha, self._beta))

            l = [x - random() * step for x in rawParam]
            r = [x + step for x in rawParam]

            for jj in xrange(100):
                rawParamNew = [l[x] + random() * (r[x] - l[x]) for x in xrange(len(rawParam))]
                trial_alpha, trial_lambda = [exp(x) for x in rawParamNew]
                lp_test = self.compute_likelihood(trial_alpha, trial_lambda)
                #print("TRYING: %f (need %f) at (%f, %f)" % (lp_test - lp_old, lp_new - lp_old, trial_alpha, trial_lambda))

                if lp_test > lp_new:
                    print(jj)
                    self._alpha = exp(rawParamNew[0])
                    self._beta = exp(rawParamNew[1])
                    self._alpha_sum = self._alpha * self._K
                    self._beta_sum = self._beta * self._V
                    rawParam = [log(self._alpha), log(self._beta)]
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
        #assert len(alpha)==self._K
        assert len(self._doc_topics)==self._D
        
        alpha_sum = alpha * self._K
        beta_sum = beta * self._V

        likelihood = 0.0
        # compute the log likelihood of the document
        likelihood += gammaln(alpha_sum) * len(self._data)
        likelihood -= gammaln(alpha) * self._K * len(self._data)
           
        for ii in self._doc_topics.keys():
            for jj in xrange(self._K):
                likelihood += gammaln(alpha + self._doc_topics[ii][jj])                    
            likelihood -= gammaln(alpha_sum + self._doc_topics[ii].N())
            
        # compute the log likelihood of the topic
        likelihood += gammaln(beta_sum) * self._K
        likelihood -= gammaln(beta) * self._V * self._K
            
        for ii in self._topic_words.keys():
            for jj in self._vocab:
                likelihood += gammaln(beta + self._topic_words[ii][jj])
            likelihood -= gammaln(beta_sum + self._topic_words[ii].N())
            
        return likelihood

    """
    compute the conditional distribution
    @param doc: doc id
    @param word: word id
    @param topic: topic id  
    @return: the probability value of the topic for that word in that document
    """
    def prob(self, doc, word, topic):
        #val = log(self._doc_topics[doc][topic] + self._alpha[topic])
        val = log(self._doc_topics[doc][topic] + self._alpha)
        #this is constant across a document, so we don't need to compute this term
        # val -= log(self._doc_topics[doc].N() + self._alpha_sum)
        
        val += log(self._topic_words[topic][word] + self._beta)
        val -= log(self._topic_words[topic].N() + self._V * self._beta)
    
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
        probs = [self.prob(doc, self._data[doc][position], x) for x in xrange(self._K)]

        #sample a new topic out of a distribution according to probs
        new_topic = log_sample(probs)

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
    def sample(self, hyper_delay=50):
        assert self._topic_assignment
        
        #sample the total corpus
        for iter in xrange(self._maximum_iteration):
            #sample every document
            for doc in self._data:
                #sample every position
                for position in xrange(len(self._data[doc])):
                    self.sample_word(doc, position)
                    
            print("iteration %i %f" % (iter, self.compute_likelihood(self._alpha, self._beta)))
            #if hyper_delay >= 0 and iter % hyper_delay == 0:
            #    self.optimize_hyperparameters()

    def print_topics(self, num_words=15):
        for ii in self._topic_words:
            print("%i:%s\n" % (ii, "\t".join(self._topic_words[ii].keys()[:num_words])))

if __name__ == "__main__":
    #d = create_data("/nfshomes/jbg/sentop/topicmod/data/de_news/txt/*.en.txt", doc_limit=50, delimiter="<doc")
    from io.de_news_io import parse_de_news_gs
    d = parse_de_news_gs("../../data/de-news/*.en.txt", "english", 100, 0.3, 0.0001)
    
    lda = CollapsedGibbsSampling()
    lda._initialize(d, 3)

    lda.sample(25)
    lda.print_topics()
