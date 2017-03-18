
# coding: utf-8

# In[1]:

get_ipython().magic(u'matplotlib inline')
import matplotlib.pyplot as plt
import glob
import os
from scipy.misc import *
import numpy as np
import tensorflow as tf
from scipy.io import loadmat


def visualize_dict(D, d_shape, patch_shape):
  ''' Displays all sparse dictionary patches in one image.
      args:
           D: the sparse dictionary with size patch size x number of patches.
           d_shape: a list or tuple containing the desired number of patches per 
                    dimension of the dictionary. For example, a dictionary with
                    400 patches could be viewed at 20 patches x 20 patches.
           patch_shape: a list that specifies the width and height
                        to reshape each patch to. '''

  if np.size(d_shape)==2:
    vis_d=np.zeros([d_shape[0]*patch_shape[0], d_shape[1]*patch_shape[1], 1])
    resize_shp=[patch_shape[0], patch_shape[1], 1]
  else:
    vis_d=np.zeros([d_shape[0]*patch_shape[0], d_shape[1]*patch_shape[1], 3])
    resize_shp=[patch_shape[0], patch_shape[1], 3]

  for row in range(d_shape[0]):
    for col in range(d_shape[1]):
      resized_patch=np.reshape(D[:, row*d_shape[1]+col], resize_shp)
      vis_d[row*patch_shape[0]:row*patch_shape[0]+patch_shape[0], 
            col*patch_shape[1]:col*patch_shape[1]+patch_shape[1], :]=resized_patch
  if vis_d.shape[2]==3:
    plt.imshow(vis_d)
  else:
    plt.imshow(vis_d.reshape([vis_d.shape[0], vis_d.shape[1]]))

    
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
    x=loadmat('patches.mat')
    x=x['data']
    
    drop_dict, drop_alpha=LCA(x, 1000, 70, num_dict_features=450)
    visualize_dict(drop_dict, [25, 20], [16, 16])


# In[ ]:




# In[ ]:




# In[ ]:



