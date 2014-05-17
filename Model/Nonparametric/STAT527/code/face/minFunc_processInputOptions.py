from collections import Counter
def minFunc_processInputOptions(o):
  # Constants
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

    verbose = 1
    verboseI= 1
    debug = 0
    doPlot = 0
    method = LBFGS
    cgSolve = 0
    
    o = toUpper(o)

    if o.has_key('DISPLAY'):
       if o['DISPLAY'].upper() == 0:
           verbose = 0
           verboseI = 0
       elif o['DISPLAY'].upper() == 'FINAL':
           verboseI = 0
       elif o['DISPLAY'].upper() == 'OFF':
           verbose = 0
           verboseI = 0
       elif o['DISPLAY'].upper() == 'NONE':
           verbose = 0
           verboseI = 0
       elif o['DISPLAY'].upper() == 'FULL'
           debug = 1
       else o['DISPLAY'].upper() == 'EXCESSIVE'
           debug = 1
           doPlot = 1
        
    DerivativeCheck = 0
    if o.has_key('DERIVATIVECHECK'):
       derivative = o['DERIVATIVECHECK'].upper()
       if derivative == 1
            DerivativeCheck = 1;
       elif derivative == 'ON'
            DerivativeCheck = 1;

    LS_init = 0;
    LS_type = 1;
    LS_interp = 2;
    LS_multi = 0;
    Fref = 1;
    Damped = 0;
    HessianIter = 1;
    c2 = 0.9;
    if o.has_key('METHOD'):
       m = o['METHOD'].upper();
       if m == 'TENSOR':
            method = TENSOR
       elif m == 'NEWTO':
            method = NEWTON
       elif m == 'MNEWTON':
            method = NEWTON
            HessianIter = 5
       elif m == 'PNEWTON0':
            method = NEWTON0
            cgSolve = 1
       elif m == 'NEWTON0':
            method = NEWTON0
       elif m == 'QNEWTON':
            method = QNEWTON
            Damped = 1
       elif m == 'LBFGS':
            method = LBFGS
       elif m == 'BB':
            method = BB
            LS_type = 0
            Fref = 20
       elif 'PCG':
            method = PCG
            c2 = 0.2
            LS_init = 2
       elif 'SCG':
            method = CG
            c2 = 0.2
            LS_init = 4
       elif 'CG':
            method = CG
            c2 = 0.2
            LS_init = 2
       elif 'CSD':
            method = CSD
            c2 = 0.2
            Fref = 10
            LS_init = 2
       elif 'SD':
            method = SD
            LS_init = 2

    maxFunEvals = getOpt(o,'MAXFUNEVALS',1000)
    maxIter = getOpt(o,'MAXITER',500)
    optTol = getOpt(o,'OPTTOL',1e-5)
    progTol = getOpt(o,'PROGTOL',1e-9)
    corrections = getOpt(o,'CORRECTIONS',100)
    corrections = getOpt(o,'CORR',corrections)
    c1 = getOpt(o,'C1',1e-4)
    c2 = getOpt(o,'C2',c2)
    LS_init = getOpt(o,'LS_INIT',LS_init)
    cgSolve = getOpt(o,'CGSOLVE',cgSolve)
    qnUpdate = getOpt(o,'QNUPDATE',3)
    cgUpdate = getOpt(o,'CGUPDATE',2)
    initialHessType = getOpt(o,'INITIALHESSTYPE',1)
    HessianModify = getOpt(o,'HESSIANMODIFY',0)
    Fref = getOpt(o,'FREF',Fref)
    useComplex = getOpt(o,'USECOMPLEX',0)
    numDiff = getOpt(o,'NUMDIFF',0)
    LS_saveHessianComp = getOpt(o,'LS_SAVEHESSIANCOMP',1)
    Damped = getOpt(o,'DAMPED',Damped)
    HvFunc = getOpt(o,'HVFUNC',[])
    bbType = getOpt(o,'BBTYPE',0)
    cycle = getOpt(o,'CYCLE',3)
    HessianIter = getOpt(o,'HESSIANITER',HessianIter)
    outputFcn = getOpt(o,'OUTPUTFCN',[])
    useMex = getOpt(o,'USEMEX',1)
    useNegCurv = getOpt(o,'USENEGCURV',1)
    precFunc = getOpt(o,'PRECFUNC',[])
    LS_type = getOpt(o,'LS_type',LS_type)
    LS_interp = getOpt(o,'LS_interp',LS_interp)
    LS_multi = getOpt(o,'LS_multi',LS_multi)


# Set the value of v 
def getOpt(options, opt, default):
    if options.has_key('opt'):
        if options['opt']:
            v = options['opt']
        else
            v = default
    else
        v = default
    
    return v

# Set the key in o uppercase name, and in a dictionary 
def toUpper(o):
    if len(o) > 0:
        fn = o.keys()
        for i in range(0, len(o))
            o = 
    if ~isempty(o):  
        dict((k.upper(), v) for k,v in o.iteritems())
    return o
        
    
