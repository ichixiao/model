# return s
import numpy as np
def logsumexp(x, dim) :
## Returns log(sum(exp(x),dim)) while avoiding numerical underflow.
## Default is dim = 1 (columns).
## Written by Mo Chen (mochen@ie.cuhk.edu.hk). March 2009.
    

    ## subtract the largest in each column
    # largest valus for each column
    y = np.max(x, 0)[dim-1]
    x = x - y
    s = y + np.log(np.sum(np.exp(x),axis = dim-1))
    i = np.nonzero(not math.isfinite(y))
    if (not i):
        s[i] = y[i]

    return s
  
