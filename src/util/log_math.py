from math import log, exp;
from random import random;

def log_add(log_a, log_b):
    if log_a < log_b:
        return log_b + log(1 + exp(log_a - log_b))
    else:
        return log_a + log(1 + exp(log_b - log_a))

def log_normalize(distribution):
    normalizer = reduce(log_add, distribution)
    for index in xrange(len(distribution)):
        distribution[index] -= normalizer
    return distribution

# sample a key from a dictionary using the values as probabilities (unnormalized)
# @param dist: defines a probability distribution, the probabilities need NOT to be normalized at all
# @return: a sample drawn from the given distribution
def log_sample(dist):
    cutoff = random()
    dist = log_normalize(dist)

    current = 0
    for index in xrange(len(dist)):
        current += exp(dist[index])
        if current >= cutoff:
            return index
    
    assert False, "Didn't choose anything: %f %f" % (cutoff, current)

#Returns the gamma function of xx.
#Gamma(z) = Integral(0,infinity) of t^(z-1)exp(-t) dt.
#(Adapted from: Numerical Recipies in C.)
#Usage:   lgammln(xx)
#Copied from stats.py by strang@nmr.mgh.harvard.edu
def lgammln(xx):
    coeff = [76.18009173, -86.50532033, 24.01409822, -1.231739516, 0.120858003e-2, -0.536382e-5]
    
    x = xx - 1.0
    tmp = x + 5.5
    tmp = tmp - (x + 0.5) * log(tmp)
    ser = 1.0
    for j in range(len(coeff)):
        x = x + 1
        ser = ser + coeff[j] / x
        
    return - tmp + log(2.50662827465 * ser)