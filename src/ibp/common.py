import numpy, scipy;

"""
sample standard deviation of a multivariant Gaussian distribution
@param sigma_hyper_parameter: the hyper-parameter of the gamma distribution
@param matrix: a r*c matrix drawn from a multivariant c-dimensional Gaussian distribution with zero mean and identity c*c covariance matrix
"""
def sample_sigma(sigma_hyper_parameter, matrix):
    assert(sigma_hyper_parameter!=None);
    assert(matrix!=None);
    assert(type(sigma_hyper_parameter)==tuple);
    assert(type(matrix)==numpy.ndarray);
    
    (sigma_hyper_a, sigma_hyper_b) = sigma_hyper_parameter;
    (row, column) = matrix.shape;

    # compute the posterior_shape = sigma_hyper_a + n/2, where n = self._D * self._K
    posterior_shape = sigma_hyper_a + 0.5 * row * column;
    # compute the posterior_scale = sigma_hyper_b + sum_{k} (A_k - \mu_A)(A_k - \mu_A)^\top/2
    var = 0;
    if row>=column:
        var = numpy.trace(numpy.dot(matrix.transpose(), matrix));
    else:
        var = numpy.trace(numpy.dot(matrix, matrix.transpose()));
    
    posterior_scale = 1.0/(sigma_hyper_b + var*0.5);
    tau = scipy.stats.gamma.rvs(posterior_shape,scale=posterior_scale);
    sigma_a_new = numpy.sqrt(1.0/tau);
    
    return sigma_a_new;

"""
sample alpha from conjugate posterior
"""
def sample_alpha(K, N, alpha_hyper_parameter):        
    assert(alpha_hyper_parameter!=None);
    assert(type(alpha_hyper_parameter)==tuple);

    (alpha_hyper_a, alpha_hyper_b) = alpha_hyper_parameter;
    
    posterior_shape = alpha_hyper_a + K;
    H_N = numpy.array([range(N)])+1.0;
    H_N = numpy.sum(1.0/H_N);
    posterior_scale = 1.0/(alpha_hyper_b + H_N);
    
    #posterior_shape = alpha_hyper_a + self._Z.sum();
    #posterior_scale = 1.0/(alpha_hyper_b + self._N);
 
    alpha_new = scipy.stats.gamma.rvs(posterior_shape,scale=posterior_scale);
    
    return alpha_new;

"""
center the data, i.e., subtract the mean
"""
def center_data(data):
    (N, D) = data.shape;
    data = data - numpy.tile(data.mean(axis=0), (N, 1));
    return data
