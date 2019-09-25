import os
import sys
import scipy.io
import scipy.misc
import matplotlib.pyplot as plt
from matplotlib.pyplot import imshow
from PIL import Image
from nst_utils import *
import numpy as np
import tensorflow as tf

model = load_vgg_model('./pretrained_model/imagenet-vgg-verydeep-19.mat')

def content_obj(a_C,a_G):
	_,nh,nw,nc = a_G.get_shape().as_list()
	a_C_reshape = tf.reshape(a_C,[1,nh*nw*nc])
	a_G_reshape = tf.reshape(a_G,[1,nh*nw*nc])
	
	J_content = 1/(4*nh*nw*nc)*tf.reduce_sum(tf.square(a_C_reshape-a_G_reshape))
	
	return J_content

def gram_matrix(A):
	GA=tf.matmul(A,A,transpose_b=True)
	return GA
	
def compute_layer_style_cost(a_S,a_G):
	m,nh,nw,nc = a_G.get_shape().as_list()
	a_S = tf.transpose(tf.reshape(a_S,[nh*nw,nc]))
	a_G = tf.transpose(tf.reshape(a_G,[nh*nw,nc]))
	GS = gram_matrix(a_S)
	GG = gram_matrix(a_G)
	J_style = tf.reduce_sum(tf.reduce_sum(tf.square(tf.subtract(GS,GG))))/(4*nc**2*(nh*nw)**2)
	return J_style

style_layers=[('conv1_1',0.2),('conv2_1',0.2),('conv3_1',0.2),('conv4_1',0.2),('conv5_1',0.2)]

def compute_style_cost(model,style_layers):
	J_style = 0
	for layer_name,coeff in style_layers:
		out = model[layer_name]
		a_S = sess.run(out)
		a_G = out
		
		J_style_layer = compute_layer_style_cost(a_S,a_G)
		
		J_style += coeff*J_style_layer
	
	return J_style

def total_cost(J_content,J_style,alpha = 10, beta = 40):
	J = alpha*J_content+beta*J_style
	return J
	
tf.reset_default_graph()
sess = tf.InteractiveSession()

content_image = scipy.misc.imread('./images/lotte_tower.jpg')
content_image = scipy.misc.imresize(content_image,(371,516))
content_image = reshape_and_normalize_image(content_image)

style_image = scipy.misc.imread('./images/gogh_style.jpg')
style_image = scipy.misc.imresize(style_image,(371,516))
style_image = reshape_and_normalize_image(style_image)

generated_image = generate_noise_image(content_image)
imshow(generated_image[0])	

model = load_vgg_model('./pretrained_model/imagenet-vgg-verydeep-19.mat')
sess.run(model['input'].assign(content_image))
out = model['conv4_2']
a_C = sess.run(out)
a_G = out
J_content = content_obj(a_C,a_G)

sess.run(model['input'].assign(style_image))
J_style = compute_style_cost(model,style_layers)

J = total_cost(J_content,J_style,alpha=10,beta=40)
optimizer = tf.train.AdamOptimizer(2.)
train_step = optimizer.minimize(J)

def model_nn(sess,input_image,num_iterations=200):
	sess.run(tf.global_variables_initializer())
	sess.run(model['input'].assign(input_image))
	
	for i in range(num_iterations):
		sess.run(train_step)
		
		generated_image = sess.run(model['input'])
		
		if i % 20 == 0 :
			Jt,Jc,Js = sess.run([J,J_content,J_style])
			print('Iteration '+str(i)+' :')
			print('total cost = '+str(Jt))
			print('content cost = '+str(Jc))
			print('style cost = '+str(Js))
			
			save_image('./output/Hyunseok_'+str(i)+'.png',generated_image)
			
	save_image('./output/generated_image.png',generated_image)
	return generated_image
	
model_nn(sess,generated_image)