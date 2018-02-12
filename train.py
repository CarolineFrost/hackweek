import numpy as np
import scipy.io
import tensorflow as tf
import matplotlib

#Filesystem organization
videogame = 'image1'
real = 'image1'
created = 'created'

#Initialize constants
#EXPERIMENT WITH THESE
noise = .5
beta = 5
alpha = 100
means = np.array([123.68, 116.779, 103.939]).reshape((1,1,1,3))

#Import VGG exisitng model
vgg_file = 'imagenet-vgg-verydeep-19.mat'
vgg_model = scipy.io.loadmat(vgg_file)
vgg_layers = vgg_model['layers']


def get_weights(layer_num):
    return vgg_layers[0][layer_num][0][0][0][0][0], vgg_layers[0][layer_num][0][0][0][0][1]

def get_bias(layer_num):
    return vgg_layers[0][layer_num][0][0][0][0][1]

