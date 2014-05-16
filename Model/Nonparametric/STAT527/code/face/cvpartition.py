import numpy as np

def cvpartition(N, k, randomize=False):

      items = np.arange(0, N)
      np.random.shuffle(items)
      slice = [items[i::k] for i in xrange(k)]
      
      cv = []
      training = []
      test = []
      for i in xrange(k):
         cv.append(slice[i])
         training.append(slice[i][0:0.8*(N/k)])
         test.append(slice[i][0.8*(N/k)+1:])


