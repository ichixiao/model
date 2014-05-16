#rewritting code
##GPLVM with mixture of Gaussian latent p(X)
##latent_dimension is the dimension of the latent space

##n_components is the number of mixture components

##Y: observations
##generate samples from the posterior instead of maximizing the likelihood. It works by alternatively sampling p(c|x) and p(x|c) 
import numpy as np
from math import sqrt
def gplvm_dpmix_integrate_infer(latent_dimension, n_components, Y, labels, options):
    (N, input_dimension) = np.shape(Y)
    num_iters = 10000

##Default options for hybrid monte carlo
options = {'num_iters':1, 'Tau':25, 'epsilon':0.001*sqrt(N), 'isPlot':0}

##
