# Initialize X 
# update needed :operataions for params 
import numpy as np
def gplvm_original(Y, Q, options):
    params = {}
# Y -- N observations, each is D dimensions
    (N, D) = np.shape(Y)
#centering
    Y = Y - np.tile(np.mean(Y, axis = 0), (N,1))
#initialize X by pca
  ## X and Y have the same dimension
    if D==Q:
       params['X'] = Y
    else :
  ## only consider X is lower dimension than Y, choose the first Q principal component
       U,s,V = np.linalg.svd(Y, full_matrices=False, compute_uv=False)
       params['X'] = U[:, 1:Q]

    params['log_hypers'] = {'alpha':0, 'betainv':0, 'gamma':0}
    options['useMex'] = 0
    if not(options.has_key('maxFunEvals')):
        options['maxFunEvals'] = 100
    options['Display'] = 'off'

    ###########call functions
    unarwpped_params = unwarp(params)
    minFunc
    rewrap


        
