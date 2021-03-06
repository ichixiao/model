##function [mix] = calmix(assignments,prior,params)
import numpy as np
from numpy.linalg import inv
def calmix(assignments, prior, params):
##n_components = size(assignments,2);
    n_components = assignments.shape[1]
##for z = 1:n_components
    for z in range(0, n_components):
    ##post.ns(z) = sum(assignments(:,z),1);
        ns[z] = sum(assignments[:,z], 0)
    ##post.alpha(z) = prior.alpha+post.ns(z);
        alpha[z] = prior['alpha'] + ns[z]
    ##post.rs(z) = prior.r+post.ns(z);
        rs[z] = prior['r'] + ns[z]
    ##post.nus(z) = prior.nu+post.ns(z);
        nus[z] = prior['nu'] + ns[z]
    ##if post.ns(z) > 0
        if ns[z] > 0:
        ##Xz = params.X(find(assignments(:,z)==1),:);
            Xz = np.matrix(params['X'])
            Xz = Xz[np.nonzero(assignments[:,z]),:]
        ##post.ms(z,:) = (prior.r*prior.m+sum(Xz,1))/(prior.r+post.ns(z));
            ms[z, :] = (prior['r'] * np.matrix(prior['m']) + sum(Xz, 0))/(prior['r'] + ns[z]);
        ##S = prior.S+Xz'*Xz+prior.r*(prior.m'*prior.m)-post.rs(z)*(post.ms(z,:)'*post.ms(z,:));
            S = prior['S'] + Xz.T * Xz + prior['r'] * (np.matrix(prior['m']).T * prior['m']) - rs[z] * (ms[z,:].T * ms[z, :])
        ##post.Chols(:,:,z) = cholcov(S);
            Chols[:,:,z] = cholcov(S)
        else:
        ##post.ms(z,:) = prior.m;
            ms[z,:] = prior['m']
        ##post.Chols(:,:,z) = cholcov(prior.S);
            Chols[:,:,z] = cholcov(prior['S'])
    ##end
##end
##mix.weights = (post.ns)./sum(post.ns);
    #Normalize
    weights = ns/sum(ns)
##mix.mus = post.ms;
    mus = ms
##for z = 1:n_components
    for z in range(0, n_components):
    ##C = chol(inv(post.Chols(:,:,z)'*post.Chols(:,:,z)));
        C = chol(inv(Chols[:,:,z].T * Chols[:,:,z]))
    ##mix.decomps(:,:,z) = sqrt(post.nus(z))*C;
        decomps[:,:,z] = sqrt(nus[z]) * C
    mix = {'C':C, 'decomps':decomps, 'weights':weights, 'mus':ms}

    return mix
##end
