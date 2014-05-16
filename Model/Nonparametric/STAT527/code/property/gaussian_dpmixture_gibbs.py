#Gaussia DP Mixture model Gibb sampling module
import numpy as np
def GDPMix_Gibb(X, n_components):
    
    #Set random seed to reproduce 
    #randn('state', 3)
    #rand('twister', 2)
    
    #[N, dim] = size(X)
    (nrow, ncol) = X.shape
    
   
    #Set priors for Gaussian-wishart
    prior_r = 1
    prior_nu = ncol
    prior_s = np.matlib.identity(ncol)
    prior_m = X.mean(0)
    # Cholesky-like decomposition for covariance matrix
    prior_Chol = cholcov(prior_s)


    #Dirichlet prior
    prior_alpha = 1
    
    #initialize assignments 
    assignments = np.zeros((nrow, n_components))
    ##cidx = kmeans(X, n_components)
    cidx = np.random.random_integers(1, n_components, nrow)
    for z in range(0, n_components) :
        assignments[cidx==z, z] = 1
    
    
    #calculate posteriors for Gaussian Wishart hyper-parameters
    post_ns = nanmatrix(n_components, 1)
    post_rs = nanmatrix(n_components, 1)
    post_nus = nanmatrix(n_components, 1)
    post_ms = nanmatrix(n_components, ncol)
    post_Chols = a=np.empty((ncol, ncol, n_components))
    
    for z in range(0, n_components) :
         post_ns[z] = sum(assignments[:, z])
         Xz = X[np.where(assignments[:,z]==1),:]
         post_rs[z] = prior_r + post_ns[z]
         post_nus[z] = prior_nu + post_ns[z]
         post_ms[z, :] = (prior_r * prior_m + sum(Xz))/(prior_r + post_ns[z])
         s = prior_s + Xz.T * Xz + prior_r * (prior_m.T * prior_m) - post_rs[z] * (post_ms[z,:]' * post_ms[z,:]) 
         post_Chols[:,:,z] = cholcov(s)
     

    
     n_iters = 300
     for iter in range(0, n_iters):
         

n_iters = 300;
for iter = 1:n_iters
    [L,assignments,post] = gaussian_dpmixture_gibbsstep(X,assignments,prior,post);
    Ls(iter) = L;
    if mod(iter,1) == 0
        %plot_mix_gauss_wishart(post);
        plot_gaussian_mixture(X,assignments,prior,post);
        figure(100); clf;
        plot(Ls);
        drawnow;
    end
end

end

#Initialize a Nan matrix
def nanmatrix(nrow, ncol, dtype=float):
    a = np.empty([nrow, ncol], dtype)
    a.fill(np.nan)
    return a
