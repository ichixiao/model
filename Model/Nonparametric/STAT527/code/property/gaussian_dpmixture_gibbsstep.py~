#return L, assignments, post
#cholupdate() defined outside 
import numpy as np
import scipy
import scipy.linalg as sl
import math
def gaussian_dpmixture_gibbsstep(X, assignments, prior, post):
    ##post: posteriors
    
    #N sample
    (N,dim) = np.shape(X)
    #assignments = np.zeros((N,n_components))
    n_components = np.shape(assignments,1)

    for n in range(0,N):
        #return indices of nonzero element
        cur_z = np.nonzero(assignments[n,:])

        assignments(n,cur_z) = 0
        # Nc = 0
        if post.ns(cur_z) == 1 :
            ## delete not assigned component
#high dimension matrix delete????
            post.Chols(:,:,cur_z) = np.delete(post.Chols, )
            assignments = np.delete(assignments, cur_z, 1)
            post.ms = np.delete(post.ms, cur_z, 0)
            post.ns = np.delete(post.ns, cur_z)
            post.rs = np.delete(post.rs, cur_z)
            post.nus = np.delete(post.nus, cur_z)
            post.alphas = np.delete(post.alphas)
            n_components = n_components-1
         else:
            ## omit the current sample  ??????
            #cholupdate function ???
            post.Chols(:,:,cur_z) = cholupdate(post.Chols(:,:,cur_z),sqrt(post.rs(cur_z))*post.ms(cur_z,:)','+')
            post.ms(cur_z,:) = post.ms(cur_z,:)*(prior.r+post.ns(cur_z))-X(n,:);
            post.ns(cur_z) = post.ns(cur_z)-1;
            post.rs(cur_z) = post.rs(cur_z)-1;
            post.nus(cur_z) = post.nus(cur_z)-1;
            post.ms(cur_z,:) = post.ms(cur_z,:)/(prior.r+post.ns(cur_z));
            post.Chols(:,:,cur_z) = cholupdate(post.Chols(:,:,cur_z),X(n,:)','-');
            post.Chols(:,:,cur_z) = cholupdate(post.Chols(:,:,cur_z),sqrt(post.rs(cur_z))*post.ms(cur_z,:)','-')
            post.alphas(cur_z) = post.alphas(cur_z)-1;
      
    
        ## Dirichlet-multinomial part
        # sum(assignments) = Nc
        prob = [math.log(sum(assignments)),math.log(prior.alpha)]
        ##prob = [log(post.ns(:,1)'),log(prior.alpha)];
    
        ## Gaussian-Wishart part
        for z in range(0, n_components):
           ## calculate posteriors when sample n is assigned to z
            Chol = cholupdate(post.Chols(:,:,z),sqrt(post.rs(z))*post.ms(z,:)','+');
            m = post.ms(z,:)*(prior.r+post.ns(z))+X(n,:)
            num = post.ns(z)+1
            r = post.rs(z)+1
            nu = post.nus(z)+1
            m = m/(prior.r+num)
            Chol = cholupdate(Chol,X(n,:)','+')
            Chol = cholupdate(Chol,sqrt(r)*m','-')
            ##prob(z) = prob(z) + loglikelihood(num,dim,r,nu,Chol);
            prob(z) = prob(z) + loglikelihood(dim,...
            post.ns(z),post.rs(z),post.nus(z),post.Chols(:,:,z),...
            num,r,nu,Chol)
   
        newm = (prior.r*prior.m+X[n,:])/(prior.r+1)
        newr = prior.r+1
        newnu = prior.nu+1
        newS = prior.S+np.matrix(X(n,:)).T*X(n,:)+prior.r*(np.matrix(prior.m).T*prior.m)-newr*(np.matrix(newm).T*newm)
        newChol = sl.cholesky(newS)
        ##prob(end) = prob(end) + loglikelihood(1,dim,newr,newnu,newChol);
        ##prob(end) = prob(end) + loglikelihood(dim, 1,dim,newr,newnu,newChol);
        prob(end) = prob(end) + loglikelihood(dim,0,prior.r,prior.nu,prior.Chol,...
        1,newr,newnu,newChol);
    
        ## normalize so as to sum to one
        psum = logsumexp(prob,2)
        prob = math.exp(prob-psum)

        ## sampling assignment
        newassignment = mnrnd(1,[prob(1:end-1),1-sum(prob(1:end-1))])
        if isnan(newassignment) : ## FIXME
            newassignment = np.zeros((1,n_components+1))
            [ignore,maxind]=max(prob)
            newassignment(maxind) = 1
    
        cur_z = np.nonzero(newassignment)

        if newassignment(end) == 1
            ##initialize posteriors
            assignments = [assignments,zeros(N,1)]
            assignments[n,cur_z] = 1
            post.ns[cur_z] = 1
            post.rs[cur_z] = prior.r+1
            post.nus[cur_z] = prior.nu+1
            post.ms[cur_z,:] = (prior.r*prior.m+X[n,:])/(prior.r+1)
            S = prior.S+np.matrix(X[n,:]).T*X[n,:]+prior.r*(np.matrix(prior.m).T*prior.m)-post.rs[cur_z]*(np.matrix(post.ms[cur_z,:]).T*post.ms[cur_z,:])
            post.Chols(:,:,cur_z) = sl.cholskey(S)
            post.alphas(cur_z) = prior.alpha+1
            n_components = n_components+1
        else
            ## update the hyperparameters according to sampled assignment
            assignments[n,cur_z] = 1
            post.Chols(:,:,cur_z) = cholupdate(post.Chols(:,:,cur_z),sqrt(post.rs(cur_z))*post.ms(cur_z,:)','+')
            post.ms(cur_z,:) = post.ms(cur_z,:)*(prior.r+post.ns(cur_z))+X(n,:);
            post.ns(cur_z) = post.ns(cur_z)+1;
            post.rs(cur_z) = post.rs(cur_z)+1;
            post.nus(cur_z) = post.nus(cur_z)+1;
            post.ms(cur_z,:) = post.ms(cur_z,:)/(prior.r+post.ns(cur_z));
            post.Chols(:,:,cur_z) = cholupdate(post.Chols(:,:,cur_z),X(n,:)','+');
            post.Chols(:,:,cur_z) = cholupdate(post.Chols(:,:,cur_z),sqrt(post.rs(cur_z))*post.ms(cur_z,:)','-');
            post.alphas(cur_z) = post.alphas(cur_z);
    
        ## check updates
    ##{
    ##for z = 1:n_components    
        ##fprintf('---------------\n');
        ##Xz = X(find(assignments(:,z)==1),:);
        ##post.ms(z,:)
        ##(prior.r*prior.m+sum(Xz,1))/(prior.r+post.ns(z))
        ##S = prior.S+Xz'*Xz+prior.r*(prior.m'*prior.m)-post.rs(z)*(post.ms(z,:)'*post.ms(z,:));
        ##post.Chols(:,:,z)
        ##cholcov(S)
    ##end
    


    L = 0
    ## likelihood for Dirichlet process
    L = L+n_components*log(prior.alpha)
    for z in range(0, n_components):
        L = L+gammaln(post.ns(z))

    L = L-gammaln(prior.alpha+N)
    L = L+gammaln(prior.alpha)

    ## likelihood for Gaussian-Wishart
    for z in range(0, n_components):
        L = L - 0.5*post.ns(z)*dim*math.log(math.pi)
        L = L - 0.5*dim*math.log(post.rs[z])
        L = L - post.nus[z]*sum(log(diag(post.Chols(:,:,z))))
        for d = 1:dim
           L = L + gammaln(0.5*(post.nus[z]+1-d))



# return L
def loglikelihood(dim,num0,r0,nu0,Chol0,num,r,nu,Chol):
    L = 0.0
    L = L-0.5*num*dim*math.log(math.pi)
    L = L+0.5*num0*dim*math.log(math.pi)
    L = L-0.5*dim*math.log(r)
    L = L+0.5*dim*math.log(r0)
    L = L-nu*sum(log(diag(Chol)))
    L = L+nu0*sum(log(diag(Chol0)))
global gammaterm_n;
    if isnan(gammaterm_n(num))
        gammaterm_n[num] = 0
        for d = 1:dim
            gammaterm_n[num] = gammaterm_n(num)+gammaln(0.5*(nu+1-d))
            gammaterm_n[num] = gammaterm_n(num)-gammaln(0.5*(nu0+1-d))
    
    L = L+gammaterm_n(num)


# return L
def loglikelihood2(dim,num0,r0,nu0,Chol0,num,r,nu,Chol):
    L = 0.0
    L = L-0.5*num*dim*math.log(math.pi)
    L = L+0.5*num0*dim*math.log(math.pi)
    L = L-0.5*dim*math.log(r)
    L = L+0.5*dim*math.log(r0)
    L = L-nu*sum(math.log(np.diag(Chol)))
    L = L+nu0*sum(math.log(np.diag(Chol0)))
    for d in range(0,dim):
        L = L+gammaln(0.5*(nu+1-d))
        L = L-gammaln(0.5*(nu0+1-d))

