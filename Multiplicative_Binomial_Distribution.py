"""
For the class of MultiplicativeBinomial distribution object and all useful functions relating to it.
"""
import numpy as np
from scipy.special import comb
from scipy.optimize import minimize

class MultiplicativeBinomial(object):
    def __init__(self, p, theta, m):
        """
        Creates the Conway-Maxwell binomial distribution with parameters p, nu, and m. Calculates the normalising function during initialisation. Uses exponents and logs to avoid overflow.
        Arguments:  self,
                    p, real 0 <= p <= 1, probability of success
                    theta, real, dispersion parameter
                    m, number of trials
        Returns:    object
        """
        self.p = p
        self.theta = theta
        self.m = m
        self.normaliser = self.getNormaliser()
        self.has_samp_des_dict = False
        self.samp_des_dict, self.has_samp_des_dict = self.getSamplingDesignDict()

    def pmf_atomic(self, k):
        """
        Probability mass function. Uses exponents and logs to avoid overflow.
        Arguments:  self, ConwayMaxwellBinomial object,
                    k, int, must be an integer in the interval [0, m]
        Returns:    P(k)
        """
        if (k > self.m) | (k != int(k)) | (k < 0):
            raise ValueError("k must be an integer between 0 and m, inclusive")
        if self.p == 1:
            p_k = 1 if k == self.m else 0
        elif self.p == 0:
            p_k = 1 if k == 0 else 0
        elif self.has_samp_des_dict:
            p_k = self.samp_des_dict.get(k)
        else:
            p_k = self.getProbMassForCount(k)/self.normaliser
        return p_k

    def pmf(self, k):
        """
        Probability mass function that can take lists or atomics.
        Arguments:  self, ConwayMaxwellBinomial object,
                    k, int, or list of ints
        Returns:    P(k)
        """
        if np.isscalar(k):
            return self.pmf_atomic(k)
        else:
            return np.array([self.pmf_atomic(k_i) for k_i in k])

    def logpmf(self, k):
        """
        Log probability mass function. Does what it says on the tin. 
        Improvement might be possible, later.
        Arguments:  self, ConwayMaxwellBinomial object,
                    k, int, must be an integer in the interval [0,m]
        Returns:    log P(k)
        """
        return np.log(self.pmf(k))

    def cdf_atomic(self, k):
        """
        For getting the cumulative distribution function of the distribution at k.
        Arguments:  self, the distribution object
                    k, int, must be an integer in the interval [0,m]
        Returns:    float

        NB: this function relies on the sampling design dictionary keys being sorted!
        """
        accumulated_density = 0
        if (k > self.m) | (k != int(k)) | (k < 0):
            raise ValueError("k must be an integer between 0 and m, inclusive")
        elif k == 0:
            return self.samp_des_dict[0]
        elif k == self.m:
            return 1.0
        else:
            for dk,dv in self.samp_des_dict.items():
                if dk <= k:
                    accumulated_density += dv
                else:
                    return accumulated_density # avoids looping through all the keys unnecessarily.
    
    def cdf(self, k):
        """
        For getting the cumulative distribution function at k, or a list of k.
        Arguments:  self, the distribution object
                    k, int, must be an integer in the interval [0,m]
        Returns:    float or array of floats
        """
        if np.isscalar(k):
            return self.cdf_atomic(k)
        else:
            return np.array([self.cdf_atomic(k_i) for k_i in k])

    def getSamplingDesignDict(self):
        """
        Returns a dictionary representing the sampling design of the distribution. That is, samp_des_dict[k] = pmf(k)
        Arguments:  self, the distribution object,
        Returns:    samp_des_dict, dictionary, int => float
                    has_samp_des_dict, True
        """
        possible_values = range(0,self.m+1)
        samp_des_dict = dict(zip(possible_values, self.pmf(possible_values)))
        has_samp_des_dict = True
        return samp_des_dict, has_samp_des_dict

    def rvs(self, size=1):
        return np.random.choice(range(0,self.m + 1), size=size, replace=True, p=list(self.samp_des_dict.values()))

    def getNormaliser(self):
        """
        For calculating the normalising factor of the distribution.
        Arguments:  self, the distribution object
        Returns:    the value of the normalising factor S(p,nu)
        """
        if (self.p == 0) | (self.p == 1):
            warnings.warn("p = " + str(self.p) + " The distribution is deterministic.")
            return 0
        else:
            return np.sum([self.getProbMassForCount(k) for k in range(0, self.m + 1)])

    def getProbMassForCount(self, k):
        """
        For calculating the unnormalised probability mass for an individual count.
        Arguments:  self, the distribution object
                    k, int, must be an integer in the interval [0, m]
        Returns:    float, 
        """
        return np.exp(np.log(comb(self.m, k)) + (k * np.log(self.p)) + ((self.m - k) * np.log(1-self.p)) + (k * (self.m - k) * np.log(self.theta)))

def multiplicativeBinomialNegLogLike(params, m, samples):
    """
    For calculating the negative log likelihood at p,theta.
    Arguments:  params: p, 0 <= p <= 1
                        theta, float, dispersion parameter
                m, number of bernoulli variables
                samples, ints between 0 and m, data.
    Returns:    float, negative log likelihood
    """
    p, theta = params
    if (p == 1) | (p == 0):
        return np.infty
    n = samples.size
    multi_bin_dist = MultiplicativeBinomial(p, theta, m)
    p_part = np.log(p/(1-p))*samples.sum()
    theta_part = np.log(theta) * np.sum((samples*(m-samples)))
    partition_part = np.log(mulit_bin_dist.normaliser) - (m * np.log(1-p)) - np.log(comb(m,samples)).sum()
    return n*partition_part - p_part - nu_part

def estimateParams(m, samples, init):
    """
    For estimating the parameters of the Conway-Maxwell binomial distribution from the given samples.
    Arguments:  m, the number of bernoulli variables being used.
                samples, ints, between 0 and m
                init, initial guess for the parameters, p and theta
    Return:     the fitted params, p and theta
    """
    bnds = ((np.finfo(float).resolution, 1 - np.finfo(float).resolution),(None,None))
    res = minimize(multiplicativeBinomialNegLogLike, init, args=(m,samples), bounds=bnds)
    return res.x

