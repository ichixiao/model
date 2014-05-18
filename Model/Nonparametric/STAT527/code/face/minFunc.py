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
# - 'pnewton0': Preconditioned Hessian-Free Newton
#              numerically computes Hessian-Vector products, preconditioned version
# - 'qnewton': Quasi-Newton Hessian approximation
#             use dense Hessian approximation
# - 'mnewton': Newton's method with Hessian calculation after every user-specified number of iterations
#             needs user-sipplied Hessian matrix
# - 'newton': Newton's method with Hessian calculation every iteration
#            needs user-supplied Hessian matrix
# - 'tensor': Tensor
#            needs user-supplied Hessian matrix and Tensor of 3rd partial derivatives
#
# Several line search strategies are available for finding a step length satisfying 
# the termination criteria ('LS_type')
# - 0 : A backtracking line-search based on the Armijo condition (default for 'bb')
# - 1 : A backeting line-search based on the strong Wolfe conditions (default for all other)
# - 2 : The line-search from the Matlab Optimization Toolbox
#
# For thr Armijo line-search, several inerpolation strategies are available ('LS_interp')
# - 0 : step size halving
# - 1 : polynomial interpolation using new function values
# - 2 : polynomial interpolation using new function and gradient values
#
# When LS_interp = 1, the default setting of LS_multi = 0 uses quadratic quadratic interpolation,
# while if (LS_multi = 1) it uses cubic interpolation if more than one point are available.
#
# When LS_interp = 2, the default setting of LS_multi = 0 uses cubic interpolation, 
# While if LS_multi = 1 is uses quartic or quintic interpolation if more than one point are available
#
# For the Wolfe line-search, these interpolation strategies are available ('LS_intrp')
# - 0 : step size doubling and bisection
# - 1 : cubic interpolation/extrapolation using new function and gradient values (default)
# - 2 : mixed quadratic/cubic interpolation/extrapolation
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
        (f, g) = funObj(x, varargin{:})
        computeHessian = 0
    else :
        (f, g) = funObj(x, varargin{:})
        computeHessian = 1
    funEvals = 1

## Derivative Check
    if checkGrad:
        if numDiff:
            fo.write('Can not do derivative checking when numDiff is 1\n');
# where is derivativeCheck
        derivativeCheck(funObj, x, 1, numDiffType, vargargin{:}) ##Checks gradient
        if computeHessian:
            derivativeCheck(funObj, x, 2, numDiffType, varargin{:})

## Outpu Log
    if verboseI:
        fprintf('%10s %10s %15s %15s %15s\n','Iteration','FunEvals','Step Length','Function Val','Opt Cond');

## Compute optimality of initial point
    optCond = max(abs(g))

# nargout?
    if nargout > 3
	# Initialize Trace
        trace={'fval':f, 'funcCount':funEvals, 'optCond':optCond}

## Exit if initial point is optimal
    if optCond <= optTol:
        exitflag = 1
        msg = 'Optimality Condition below optTol\n'
        if verbose
            print msg
# nargout? output reorganize needed
        if nargout > 3
            output = {}

## Output Function
    if len(outputFcn) > 0
        stop = outputFcn()
            if stop:
                exitflag = -1
                msg = 'Stopped by output function'
                if verbose:
                    print msg
# nargout? output reorganize needed
                if nargout > 3
                    output = {}
## Perform up to maximum of 'maxIter' descent steps














