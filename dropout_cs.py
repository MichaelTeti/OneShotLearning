import tensorflow as tf
import numpy as np
from scipy.misc import *
from scipy.fftpack import idct

im=imresize(imread('lena.jpg'), 0.4)

im=np.mean(im, 2)

imvec=im.reshape([im.size, 1])

imvec=(imvec-np.mean(imvec))/np.std(imvec)

n=len(imvec)

phi=np.random.randn(n/2, n)

y=np.matmul(phi, imvec)

psi=idct(np.identity(n))

D=np.matmul(phi, psi)

lambda=2

t=1

h=0.0001

d=h/t

u=np.zeros([n, 1])

for i in range(2000):

	a=(u-tf.sign(u)*lamb)*(tf.cast(tf.abs(u)>lamb, tf.float32)) 

	u=u+h*(tf.matmul(tf.transpose(D), (y-tf.matmul(D, a)))-u-a) 
