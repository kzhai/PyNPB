#!/usr/bin/pyhon

# Original Author: Jordan Boyd-Graber
# Email: jbg@umiacs.umd.edu

# Modification: Ke Zhai
# Email: zhaike@cs.umd.edu

from collections import defaultdict
from math import log, exp
from random import random
from nltk import FreqDist
from scipy.special import gamma, psi, gammaln, polygamma
from util.log_math import log_sample

class CollapsedGibbsSampling:
    def __init__(self):
        self._docs = defaultdict(FreqDist)
        self._topics = defaultdict(FreqDist)
        
        self._state = None

        # set the document smooth factor
        self._alpha = 0.1
        # set the vocabulary smooth factor
        self._beta = 0.01
        
        self._alpha_update_decay_factor = 0.9
        self._alpha_maximum_decay = 10
        
        self._alpha_converge = 0.000001
        self._alpha_maximum_iteration = 100
        
        self._maximum_iteration = 200
        self._converge = 0.00001

        # pending for further changing~
        self._gamma_converge = 0.000001
        self._gamma_maximum_iteration = 400
        
    # data: a dict data type, indexed by document id, value is a list of words in that document, not necessarily be unique
    def _initialize(self, data, num_topics=10):
        self._K = num_topics
        
        self._alpha_sum = self._alpha * self._K
        self._state = defaultdict(dict)
    
        self._data = data
        
        self._D = len(data)
    
        # initialize the vocabulary, i.e. a list of distinct tokens.
        self._vocab = set([])
        for doc in self._data:
            for position in xrange(len(self._data[doc])):
                # learn all the words we'll see
                self._vocab.add(self._data[doc][position])
            
                # initialize the state to unassigned
                self._state[doc][position] = -1
        self._V = len(self._vocab)
        
        #self._beta_sum = float(self._V) * self._beta

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

    # compute the log-likelihood of the model
    def compute_likelihood(self, alpha, beta):
        alpha_sum = alpha * self._K
        beta_sum = beta * self._V

        likelihood = 0.0
        likelihood += gamma(alpha_sum) * len(self._docs)
        likelihood -= gamma(alpha) * self._K * len(self._docs)
        for ii in self._docs:
            for jj in xrange(self._K):
                likelihood += gamma(alpha + self._docs[ii][jj])
            likelihood -= gamma(alpha_sum + self._docs[ii].N())
      
            likelihood += gamma(beta_sum) * self._K
            likelihood -= gamma(beta) * self._V * self._K
            for ii in self._topics:
                for jj in self._vocab:
                    likelihood += gamma(beta + self._topics[ii][jj])
                likelihood -= gamma(beta_sum + self._topics[ii].N())
            
        return likelihood

    # compute the conditional distribution
    def prob(self, doc, word, topic):
        val = log(self._docs[doc][topic] + self._alpha)
        # this is constant across a document, so we don't need to compute this term
        # val -= log(self._docs[doc].N() + self._alpha_sum)
        
        val += log(self._topics[topic][word] + self._beta)
        val -= log(self._topics[topic].N() + self._V * self._beta)
    
        return val

    # this method samples the word at position in document, by covering that word and compute its new topic distribution
    # doc: a document id
    # position: the position in doc, ranged as range(self._data[doc])
    # in the end, both self._state, self._docs and self._topics will change
    def sample_word(self, doc, position):
        # retrieve the word
        word = self._data[doc][position]
    
        # get the old topic assignment to the word in doc at position
        old_topic = self._state[doc][position]
        if old_topic != -1:
            # this word already has a valid topic assignment, decrease the topic|doc counts and word|topic counts by covering up that word
            self.change_count(doc, word, old_topic, -1)
    
        probs = [self.prob(doc, self._data[doc][position], x) for x in xrange(self._K)]

        # sample a new topic out of a distribution according to probs
        new_topic = log_sample(probs)

        self.change_count(doc, word, new_topic, 1)
        self._state[doc][position] = new_topic

    # this methods change the count of topic|doc and word|topic by delta
    # this values will be used in the computation
    def change_count(self, doc, word, topic, delta):
        self._docs[doc].inc(topic, delta)
        self._topics[topic].inc(word, delta)

    # sample the corpus to train the parameters
    def sample(self, hyper_delay=50):
        assert self._state
        for iter in xrange(self._maximum_iteration):
            for doc in self._data:
                for position in xrange(len(self._data[doc])):
                    self.sample_word(doc, position)
                    
            print("Iteration %i %f" % (iter, self.compute_likelihood(self._alpha, self._beta)))
            if hyper_delay >= 0 and iter % hyper_delay == 0:
                self.optimize_hyperparameters()

    def print_topics(self, num_words=15):
        for ii in self._topics:
            print("%i:%s\n" % (ii, "\t".join(self._topics[ii].keys()[:num_words])))

if __name__ == "__main__":
    #d = create_data("/nfshomes/jbg/sentop/topicmod/data/de_news/txt/*.en.txt", doc_limit=50, delimiter="<doc")
    from io.de_news_io import parse_de_news_gs
    d = parse_de_news_gs("/windows/d/Data/de-news/txt/*.en.txt", "english", 100, 0.3, 0.0001)
    
    lda = CollapsedGibbsSampling()
    lda._initialize(d)

    lda.sample(25)
    lda.print_topics()
