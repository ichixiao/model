##function [params, nll, arate] = hmc(likefunc, x, options, labels, varargin)     
## Hamiltonian Monte Carlo
##
## David Duvenaud
## Tomoharu Iwata
##
## April 2012
##
## likefunc returns nll, dnll
##
## options.Tau is the number of leapfrog steps. 
## options.epsilon is step length

##{
##if options.isPlot == 1
##   assignments = varargin{end-2};
##    params = varargin{end-1};
##    prior = varargin{end};
##    mix = calmix(assignments,prior,params);
##end
##%}
import numpy as np
def hmc(likefunc, x, options, labels, varargin):
    arate = 0 #acceptance rate
    L = options['num_iters']
    #Call likefunc here
    [E, g] = likefunc( x, varargin{:})

##for l = 1:L
    for l in range(0, L):
    ##p = randn( size( x ) );
    ##H = p' * p / 2 + E;
        p = numpy.random.randn(np.shape(X))
        H = p.T * p/2 + E
        xnew = x 
        gnew = g
 
        cur_tau = np/random.random((options['Tau'],))
        cur_eps = np.random.uniform(0,1,1) * options['epsilon']
    ##cur_tau = options.Tau;
    ##cur_eps = options.epsilon;
    ##for tau = 1:cur_tau
        for tau in range(0, cur_tau):
            p = p - cur_eps * gnew / 2
            xnew = xnew + cur_eps * p;
            [ignore, gnew] = likefunc( xnew, varargin{:})
        
            p = p - cur_eps * gnew / 2;
    ##end
    
        [Enew, ignore] = likefunc( xnew, varargin{:});    
        Hnew = p.T * p / 2 + Enew
        dh = Hnew - H
    
        if dh < 0:
            accept = 1
            fprintf('a')
        else:
            if rand() < exp(-dh):
               accept = 1
               fprintf('A')
            else
               accept = 0
               fprintf('r')
        ##end
    ##end
    
        if accept:
            g = gnew
            x = xnew
            E = Enew
            arate = arate+1
    ##end
##end
 
    arate = arate/L
    params = x
    nll = E
