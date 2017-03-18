import tensorflow as tf
from scipy.misc import *
import numpy as np
from image_reader import *
from skimage.util import view_as_windows as vaw


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
    #a=a*tf.cast(tf.round(tf.random_uniform([num_dict_features, batch_sz])), tf.float64)
    D=D+tf.matmul((batch-tf.matmul(D, a)), tf.transpose(a))
  return sess.run(D), sess.run(a)



with tf.Session() as sess:

  im=imresize(imread('lily.jpg'), [100, 100])

  recd=np.zeros(im.shape)

  recnd=np.zeros(im.shape)

  testim=vaw(im, (ps, ps, 3))

  testim=testim.reshape([testim.shape[0]*
			 testim.shape[1]*
			 testim.shape[2], -1])

  testim=normalize(testim.transpose())

  d=np.load('dropout.npy')

  nd=np.load('no_dropout.npy')
 
  dict_, alphad=LCA(testim, 1, testim.shape[1], D=d)

  dict_, alphand=LCA(testim, 1, testim.shape[1], D=nd)

  newd=np.matmul(d, alphad)

  newnd=np.matmul(nd, alphand)

  for i in xrange(8, recd.shape[0]-7):

    for j in xrange(8, recd.shape[1]-7):

      dpatch=newd[:, (i-8)*(recd.shape[1]-7-8)+(j-8)].reshape([16, 16, 3])

      ndpatch=newnd[:, (i-8)*(recd.shape[1]-7-8)+(j-8)].reshape([16, 16, 3])
      
      recd[i-8:i+8, j-8:j+8, :]+=dpatch

      recnd[i-8:i+8, j-8:j+8, :]+=ndpatch

  recd=recd/15
 
  recnd=recnd/15

  print(np.sum((recd-im)**2))

  print(np.sum((recnd-im)**2))

  imshow(recd)
  
  imshow(recnd)

  imshow(im)

  imshow(np.absolute(im-recd))

  imshow(np.absolute(im-recnd))


