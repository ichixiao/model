# demo of the warped mixture models
# face cluster demo
import numpy as np
import scipy.io as sio

num_fold = 1
options = {'isPlot':50, 'isMovie':0, 'hmc_isPlot':0, 'isGPLVMinit':1}
data = sio.loadmat('/umist_downsampled3.mat')
X = data['X']
(N, observed_dimension) = np.shape(X)

#rescale dataset to [-1, 1]
X = X - np.tile(X.min(axis=0), (N,1))
X = X/(X.max(axis=0))
X = X*2 - 1

#dataset partition
if num_fold > 1:
   slice = [items[i::k] for i in xrange(k)]
   #cv = []
   training = []
   test = []
   for i in xrange(k):
      #cv.append(slice[i])
      training.append(slice[i][0:0.8*(N/k)])
      test.append(slice[i][0.8*(N/k)+1:])


for k in range(0, num_fold)
    if num_fold > 1:
      trainX = X[training[k], :]
      testX = X[test[k], :]
      trainy = y[training[k]]
      testy = y[test[k]]
    else :
      trainX = X
      testX = []
      trainy = y


    latent_dimensions = 2
    options_update = {'num_iters':500, 'epsilon':0.005, 'Tau':25, 'prior_r':1e-1, 'prior_alpha':1}
    options.update(options_update)
    
    num_components = 5

#infinite warped mixture model
    options_update1 = {'isDP':1, 'isGP':1}
    for j in range(0, latent_dimension):
      call gplvm_dpmix_integrate_infer
    




