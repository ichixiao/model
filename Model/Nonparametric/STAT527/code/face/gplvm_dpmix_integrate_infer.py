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
    hmc_options = {'num_iters':1, 'Tau':25, 'epsilon':0.001*sqrt(N), 'isPlot':0}

    ##Default model options
    isDP = 1
    isGP = 1
    isSamplingDPparameter = 0
    isPlot = 1
    isFixedGauss = 0
    prior_r = 1
    prior_alpha = 1

    ##options from parameter
    if options.has_key('prior_r'):
        prior_r = options['prior_r']

    if options.has_key('prior_alpha'):
        prior_alpha = options['prior_alpha']

    if options.has_key('isMovie'):
        isMovie = options['isMovie']

    if options.has_key('isPlot'):
        isPlot = options['isPlot']

    if options.has_key('isFixedGauss'):
        isFixedGauss = options['isFixedGauss']

    if options.has_key('isSamplingDPparameter'):
        isSamplingDPparameter = options['isSamplingDPparameter']

    if options.has_key('num_iters'):
        num_iters = options['num_iters']

    if options.has_key('epsilon'):
        epsilon = options['epsilon']

    if options.has_key('Tau'):
        Tau = options['Tau']

    if options.has_key('hmc_isPlot'):
        hmc_isPlot = options['hmc_isPlot']

    if options.has_key('no_warp'):
        no_warp = options['no_warp']

    if options.has_key('isGP'):
        isGP = optionsp['isGP']

    if options.has_key('isGPLVMinit'):
        isGPLVMinit = options['isGPLVMinit']

    if options.has_key('isback'):
        isback = options['isback']

    if options.has_key('circle_size'):
        circle_size = options['circle_size']

    if options.has_key('circle_alpha'):
        circle_alpha = options['circle_alpha']

    if options.has_key('num_points'):
        numpoints = options['num_points']

    ##centering
    Y = Y - np.tile(np.mean(Y, axis=0), (N,1))

    ##Initialize X as same as the observed data Y
    if isGP==0:
       latent_dimension = input_dimension
    
    if latent_dimension == input_dimension:
       init_X = Y
    elif isGPLVMinit == 1:
       gplvm_options = {}
#########call function gplvm_original here
       gplvm_params = gplvm_original(Y, latent_dimension, gplvm_options)
       init_X = gplvm_params['X']
    elif latent_dimension <= input_dimension:
        init_X = Y[:, 0:latent_dimension]
    else :
        init_X[:,0:input_dimension] = Y
        init_X[:, input_dimension:-1] = 0

    if sum(sum(init_X)) == 0:
        init_X = init_X + np.random.randn(N, latent_dimension)

    log_hypers = {'alpha':-1, 'betainv':-1, 'gamma':-1}

    prior = {'r':prior_r, 'nu':latent_dimension, 'S':np.identity(latent_dimension), 'm':np.zeros((1,latent_dimension)), 'Chol':np.linalg.cholesky(np.identity(latent_dimension)), 'alpha':prior_alpha, 'a':1, 'b':1}

    if n_components == 0:
        n_components = 1
        assignments = np.zeros((N,1))
        assignments[1, 1] = 1
    else:
        assignments = np.zeros((N, n_components))
######kmeans
        cidx = np.array()
        for z in range(0, n_components):
            assignments[np.where(cidx==z),z] = 1

    post = {'ns':np.empty([n_components,1]),
            'rs':np.empty([n_components,1]),
            'nus':np.empty([n_components,1]),
            'ms':np.empty([n_components, latent_dimension]),
            'Chols':[np.empty([latent_dimension, latent_dimension])]*n_component,
            'alphas':np_empty([n_components,1])}

    params = {'X':init_X, 'log_hypers':log_hypers}
    gammaterm_n = np.empty([N,1])
    acceptance_rate = 0
    arate_cnt = 0
    arate_start = 100

    for i in range(0, num_iters):
        if isFixedGauss == 0:
            for z in range(0, n_components):
                post['ns'][z] = sum(assignments[:,z])
                post['alpha'][z] = prior['alpha'] +post['ns'][z]
                post['rs'][z] = prior['r'] + post['ns'][z]
                post['nus'][z] = prior['nu'] + post['ns'][z]
                if post['ns'][z] > 0:
                    Xz = params[X][np.where(assignments[:,z]==1)[1],:]
                    post['ms'][z,:] = (prior['r']*prior['m'] + sum(Xz, axis=0))/(prior['r'] + post['ns'][z])
                    S = prior['S'] + Xz.T * Xz + prior['r']*(np.matrix(prior['m']).T * prior['m']) - post['rs'][z]*(post['ms'][z,:].T*post['ms'][z,:])
                    post['Chols'][z] = np.linalg.cholesky(prior['S'])
                else:
                    post['ms'][z,:] = prior['m']
                    post['Chols'][z] = np;linalg.cholesky(prior['S'])

            if isGP == 1:
                (L, assignments, post) = gaussian_dpmixture_gibbsstep(params['X'], assignments, prior, post)

            n_components = assignments.shape[1]
            Ls[i] = L
            indice = np.where(sum(assignments, axis=0)>0)[1]
            Ks[i] = len(indice)

            if isSamplingDPparameter == 1:
                prior['alpha'] = sampling_dphyperparameter(prior['alpha'], N, n_components, prior['a'], prior['b'])

        if isGP == 1:
            unwrapped_params = unwrap(params)
 

