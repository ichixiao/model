import numpy as np
import scipy
import scipy.linalg as sl
def gaussian_dpmixture_gibbs(X,n_components):

    ## post: posteriors
    ## Set the random seed, always the same for the datafolds.
# Set seed in python    
    ##randn('state', 3);
    ##rand('twister', 2);    
    ##addpath('util');
    

    (N,dim) = np.shape(X)

    ##Set priors for Gaussian-Wishert
    prior.r = 1
    prior.nu = dim
    prior.S = np.eye(dim)
    #prior.m = np.zeros((1,dim))
    #mean of columns
    prior.m = np.mean(X,axis = 0)
    #cholcov decomposition for the prior matrix
    prior.Chol = sl.cholesky(prior.S)

    ##Dirichlet prior
    prior.alpha = 1
    
    ##initialize assigments with kmeans
    assignments = np.zeros((N,n_components))
    ##cidx = kmeans(X,n_components)
    cidx = np.random.random_integers(1, n_components,N) - 1
    for z in range(0, n_components):
        assignments[cidx==z,z] = 1

    ##calculate posteriors for Gaussian-Wishart hyper-parameters
    post.ns = np.empty((n_components,1))
    post.rs = np.empty((n_components,1))
    post.nus = np.empty((n_components,1))
    post.ms = np.empty((n_components,dim))
    post.Chols = np.exmpty((dim,dim,n_components))
    for z in range(0, n_components):
        #total number z component assignment
        #Nc
        post.ns[z] = sum(assignments[:,z],axis = 0)
        #find() return the sample index which assigned to the z component
        Xz = np.matrix(X(find(assignments[:,z]==1),:))
        # r = r + Nc
        post.rs[z] = prior.r+post.ns[z]
        # v = v + Nc
        post.nus[z] = prior.nu+post.ns[z]
        # posterior mean
        post.ms[z,:] = (prior.r * prior.m+np.sum(Xz,axis=0))/(prior.r+post.ns[z])
        # dim * dim matrix
        S = prior.S+Xz.T*Xz+prior.r*(np.matrix(prior.m).T*prior.m)-post.rs[z]*(np.matrix(post.ms[z,:]).T*post.ms[z,:])
        post.Chols[:,:,z] = sl.cholesky(S)

####still need correction below
    n_iters = 300;
    for iter in range(0, n_iters):
#return multiple values in python
        [L,assignments,post] = gaussian_dpmixture_gibbsstep(X,assignments,prior,post);
        Ls(iter) = L
        if mod(iter,1) == 0
            %plot_mix_gauss_wishart(post);
            plot_gaussian_mixture(X,assignments,prior,post);
            figure(100); clf;
            plot(Ls);
            drawnow;
    



function plot_gaussian_mixture(X,assignments,prior,post)

figure(123423); clf;

[~,n_components] = size(assignments);

for z = 1:n_components
    plot(X(find(assignments(:,z)==1),1),...
         X(find(assignments(:,z)==1),2),...
         'x','Color',colorbrew(z)); hold on;
end

mix.mus = post.ms;
for z = 1:n_components
    C = cholcov(inv(post.Chols(:,:,z)'*post.Chols(:,:,z)));
    mix.decomps(:,:,z) = sqrt(post.nus(z))*C;
end
%mix.weights = (post.ns+prior.alpha)./sum(post.ns+prior.alpha,1);
mix.weights = (post.ns)./sum(post.ns);

[xmin,xmax,ymin,ymax] = plot_contours( mix );
xlim( [xmin xmax] );
ylim( [ymin ymax] );
set(gcf, 'color', 'white');
drawnow;

end

