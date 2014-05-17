import numpy as np
def minFunc(funObj, x0, options, varargin):
# unconstrained optimizer using a line search strategy
# compute descent directions using one of ('Method'):
# -'sd': Steepest Descent
#       no previouse information, not recommended
# - 'csd': Cyclic Steepest Descent
#         uses previous step length for a fixed length cycle
# - 'bb': Barzilai and Borwein Gradient
#        uses only previous step 
# - 'cg': Non-linear Conjugate Gradient
#        uses only previous step and a vector beta
# - 'scg': Scaled Non-linear Conjugate Gradient
#         uses previous step and a vector beta, and Hessian-vector products to initialize line search
# - 'pcg': Preconditioned Non-linear Conjugate Gradient 
#         uses only previous step and a vector beta, preconditioned version
# - 'lbfgs': Quasi-Newton with Limited-Memory BFGS Updating
#           default: uses a predetermined number of previous steps to form a low-rank Hessian approximation
# - 'newton0': Hessian-Free Newton
#             numerically computes Hessian-Vector products
# - 

## Constants
    SD = 0
    CSD = 1
    BB = 2
    CG = 3
    PCG = 4
    LBFGS = 5
    QNEWTON = 6
    NEWTON0 = 7
    NEWTON = 8
    TENSOR = 9
## Initialize
    p = len(x0) 
    d = np.zeros((p,1))
    x = x0
    t = 1

## If necessary, form numerical differentiation functions
    if useComplex:
        numDiffType = 3
    else:
        numDiffType = numDiff

    if (useComplex and method != TENSOR):   
        varargin.insert(0, funObj)
        varargin.insert(0, numDiffType)
        if method != NEWTON
            if debug:
                if useComplex:
                    fo.write('Using complex differentials for gradient computation\n');
	        else:
                    fo.write('Using finite differences for gradient computation\n'); 
# where is autoGrad ?
            funObj = autoGrad
        else:
            if debug:
                if useComplex:
                    fo.write('Using complex differentials for Hessian computation\n');
                else:
                    fo.write('Using finite differences for Hessian computation\n');
# where is autoHess
             funObj = autoHess
      
        if (useComplex == 1 and method == NEWTON0):
            if debug:
                fo.write('Turning off the use of complex differentials for Hessian-vector product\n');
            useCOmplex = 0
        if useComplex:
            funEvalMultiplier = p
        elif numDiff == 2:
            funEvalMultiplier = 2*p
        else:
            funEvalMultiplier = p+1

## Evaluate Initial Point
    if method < NEWTON:
        (f, g) = funObj












