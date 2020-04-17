# Multiplicative Binomial Distribution
 This package defines a class of Multiplicative-Binomial distribution objects for Python 3. It also defines some useful functions for working with the objects.

#### Requirements
- Python3.6 or above
- numpy
- scipy

## Multiplicative binomial distributions
 If you would like to learn more about this probability distribution, see this paper: <https://arxiv.org/pdf/1404.1856v1.pdf>

## Initialising a Multiplicative binomial distribution
 To initialse a multiplicative binomial distribution use the following lines of code
 ```python
 import MultiplicativeBinomial as mbin
 p = 0.4
 theta = 0.9
 m = 100
 multi_bin_distn = mbin.MultiplicativeBinomial(p, theta, m)
 ```
 To sample from the distribution run
 ```python
 multi_bin_distn.rvs(size=10)
 ```
 and to evaluate the probability mass function for some outcome, run
 ```python
 multi_bin_distn.pmf(99)
 multi_bin_distn.pmf(40)
 ```

## Estimating the parameters of the distribution
 To estimate the parameters of the Multiplicative binomial distribution given a sample, run the following lines
 ```python
 sample = multi_bin_distn.rvs(size=15)
 initial_params = [0.5, 1]
 mbin.estimateParams(m, sample, initial_params)
 ```
 To evaluate the negative log-likelihood of a sample, run
 ```python
 mbin.MultiplicativeBinomialNegLogLike([p, nu], m, sample)
 ```

 TODO: There are issues here. Calculating the partition function results in overflow.
