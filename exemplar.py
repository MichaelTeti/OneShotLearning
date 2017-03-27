import glob
import os
from scipy.misc import *
import numpy as np
import tensorflow as tf
from image_reader import *
from skimage.util import view_as_windows as vaw
import sys
from numpy import genfromtxt
import h5py


ps=16
k=600
train_sz=600

    
def LCA(y, iters, batch_sz, num_dict_features=None, D=None):
  ''' Dynamical systems neural network used for sparse approximation of an
      input vector.

      Args: 
           y: input signal or vector, or multiple column vectors.

           num_dict_features: number of dictionary patches to learn for each
                              respective dictionary. A list of integers.

           iters: number of LCA iterations.

           batch_sz: number of samples to send to the network at each iteration.

           D: The pretrained dictionary(s) to be used in the network. '''

  
  assert(num_dict_features is None or D is None), 'provide D or num_dict_features, not both'

  if D is None:
    D=np.random.randn(y.shape[0], num_dict_features)

  for i in range(iters):
    batch=y[:, np.int32(np.floor(np.random.rand(batch_sz)*y.shape[1]))]
    D=tf.matmul(D, tf.diag(1/(tf.sqrt(tf.reduce_sum(D**2, 0))+1e-6)))
    a=tf.matmul(tf.transpose(D), batch)
    a=tf.matmul(a, tf.diag(1/(tf.sqrt(tf.reduce_sum(a**2, 0))+1e-6)))
    a=0.3*a**3
    #a=a*tf.round(tf.random_uniform([num_dict_features, batch_sz], dtype=tf.float64))
    D=D+tf.matmul((batch-tf.matmul(D, a)), tf.transpose(a))

  return sess.run(D), sess.run(a)



with tf.Session() as sess:

  f=h5py.File('mnist_data_labels.h5', 'r')
  train=np.asarray(f['data'])
  trainl=np.asarray(f['labels'])

  f=h5py.File('mnist_test.h5', 'r')
  data=np.asarray(f['test_data'])
  labels=np.asarray(f['test_labels'])

  f=h5py.File('dict.h5', 'r')
  dict_=np.asarray(f['d'])

  f=h5py.File('dict_nodrop.h5', 'r')
  dictnd=np.asarray(f['d'])

  r0=np.asarray(np.where(trainl==0))

  r1=np.asarray(np.where(trainl==1))

  ad1=np.zeros([k, 13*13, train_sz])

  and1=np.zeros([k, 13*13, train_sz])

  for i in range(train_sz):

    sys.stdout.write('Image %d...      \r'%(i))
    sys.stdout.flush()
    
    im=train[r1[0, i], :, :]
 
    im=vaw(im, (ps, ps))
  
    im=im.reshape([im.shape[0]*
 		   im.shape[1], -1])

    im=normalize(im).transpose()

    d, a=LCA(im, 1, im.shape[1], D=dict_)

    nd, nda=LCA(im, 1, im.shape[1], D=dict_)

    ad1[:, :, i]=a

    and1[:, :, i]=nda

  ad1=np.mean(ad1, 2)
 
  and1=np.mean(and1, 2)

  f=h5py.File('alphas1.h5', 'a')
  f.create_dataset('drop_alpha1', data=ad1)
  f.create_dataset('no_drop_alpha1', data=and1)
  f.close()
    
    




