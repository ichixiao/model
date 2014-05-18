import numpy as np
import math
def gplvm_likelihood( X, Y, log_hypers ) :
    ## Returns the likelihood of the GP-LVM, along with 
    ## derivatives w.r.t. X and kernel hyperparameters.
    
    (N, observed_dimension) = np.shape(Y)
    (latent_dimension) = np.shape(X)[1]
    
    ## Compute kernel matrix in-place (squared exponential kernel function).
    # Explicit specify the kernel function: ?
    K1 = X * X.transpose() 
    Dia = np.diag(K1) 
    K1 = K1 - np.ones((N,1)) * Dia.transpose() / 2 
    K1 = K1 - Dia * np.ones((1,N)) / 2 
    K1 = math.exp(log_hypers.gamma) * K1 
    
    K2 = math.exp(log_hypers.alpha + K1) ;
    K = K2 + np.eye(N)*max(math.exp(log_hypers.betainv), 1e-3)  # HACK

    ##{
    ##hyp[0] = -log_hypers.gamma/2
    ##hyp[1] = log_hypers.alpha/2
    ##K2 = covSEiso(hyp, X)
    ##K = K2 + eye(N)*min(max(exp(log_hypers.betainv),1e-6),1e+3);  % HACK
    ##K = K2 + eye(N)*max(exp(log_hypers.betainv), 1e-3);  % HACK
    ##}

    
    ## Calculate objective function (negative log likelihood)
    # formula specify ?
    tmp = (K \ Y) * Y.transpose()
    gradLK = 0.5 .* (tmp - observed_dimension * np.eye(N)) / K
    f = 0.5 * observed_dimension * logdet(K) + 0.5 * trace(tmp) 

    grad_X = zeros( N, latent_dimension ) 
    for n1 = 1:N
        #grad_X(n1,:) = -gradLK(n1, :)*( -2*exp(log_hypers.gamma)* ...
        #                (repmat(X(n1,:),N,1) - X) .* ...
        #                repmat(K(:,n1),1,latent_dimension)) ;
        
        #grad_X(n1,:) = -gradLK(n1, :)*((repmat(X(n1,:),N,1) - X) .* ...
        #                repmat(K(:,n1),1,latent_dimension)) ;        
        xn = X[n1,:]
        Kn = K[:,n1]
        grad_X[n1,:] = -gradLK[n1, :]*((xn(ones(N,1),:)-X).*...
                        Kn[:,ones(1,latent_dimension))]
  
    grad_X = grad_X * -2*math.exp(log_hypers.gamma)
    log_hypers_grad.alpha = -sum(sum(gradLK.*K2))
    log_hypers_grad.betainv = -math.exp(log_hypers.betainv) * trace(gradLK) 
    log_hypers_grad.gamma  = -sum(sum(gradLK.*(K1.*K2)))

    ##gradLa=0; gradLb=0; gradLg=0; #no kernel parameter update
    
    # want to return f, grad_X, log_hypers_grad
    # f: 
    # grad_X:
    #log_hypers_grad:
