import glob
import os
from scipy.misc import *
import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from image_reader import *
from skimage.util import view_as_windows as vaw
import sys


ps=16

    
def LCA(y, iters, batch_sz, num_dict_features=None, D=None):
  ''' Dynamical systems neural network used for sparse approximation of an
      input vector.

      Args: 
           y: input signal or vector, or multiple column vectors.

           num_dict_features: number of dictionary patches to learn.

           iters: number of LCA iterations.

           batch_sz: number of samples to send to the network at each iteration.

           D: The dictionary to be used in the network. 
  '''
  
  assert(num_dict_features is None or D is None), 'provide D or num_dict_features, not both'
  if D is None:
    D=np.random.randn(y.shape[0], num_dict_features)
  for i in range(iters):
    batch=y[:, np.int32(np.floor(np.random.rand(batch_sz)*y.shape[1]))]
    D=tf.matmul(D, tf.diag(1/(tf.sqrt(tf.reduce_sum(D**2, 0))+1e-6)))
    a=tf.matmul(tf.transpose(D), batch)
    a=tf.matmul(a, tf.diag(1/(tf.sqrt(tf.reduce_sum(a**2, 0))+1e-6)))
    a=0.3*a**3
    a=a*tf.cast(tf.round(tf.random_uniform([num_dict_features, batch_sz])), tf.float64)
    D=D+tf.matmul((batch-tf.matmul(D, a)), tf.transpose(a))
  return sess.run(D), sess.run(a)



with tf.Session() as sess:

  x, y=read_ims('/home/mpcr/Documents/MT/CSDL/17flowers/jpg', 100)

  x=np.pad(x[:80, :, :, :], ((0, 0), (ps/2, ps/2), (ps/2, ps/2), (0, 0)), 'edge')

  x=vaw(x, (1, ps, ps, 3))

  x=x.reshape([x.shape[0]*
	       x.shape[1]*
	       x.shape[2]*
	       x.shape[3], -1])

  x=normalize(x.transpose())

  sys.stdout.write('Learning Dictionary 1...      \r')
  sys.stdout.flush()

  d, a=LCA(x, 700, 200, num_dict_features=200)
  visualize_dict(nodropd, [20, 10, 3], [16, 16])




