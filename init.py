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

#Activation function from https://www.tensorflow.org/api_guides/python/nn#Activation_Functions
def activ_func(feat):
    return tf.nn.relu(feat)

#Convolution function from https://www.tensorflow.org/api_guides/python/nn#Convolution
#Takes in 1 layer from model to inform filter, 1 layer to act as input
def conv_func(input_num, fil_num):
    w = tf.constant(get_weights(fil_num))
    b = tf.constant(get_bias(fil_num))
    return tf.nn.conv2d(input_num, filter=w, strides=[1, 1, 1, 1], padding='SAME') + b

#Pooling function from https://www.tensorflow.org/api_guides/python/nn#Pooling
def pool():
    return tf.nn.avg_pool(prev_layer, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

#Network
#Layer configuration information here: https://gist.githubusercontent.com/ksimonyan/3785162f95cd2d5fee77/raw/f02f8769e64494bcd3d7e97d5d747ac275825721/VGG_ILSVRC_19_layers_deploy.prototxt
#Given layer -> conv2d(network layer w/ vgg layer) -> relu -> save in network
#             [ repeat with each necessary vgg layer ]
#            -> pool -> save in network
#5 pools total
def create_network(heigh, width, color_num):
    network = {}
    network['init'] = tf.Variable(np.zeros((1, height, width, color_num))*1.0)

    #1
    network['11'] = activ_func(conv_func(graph['init'], 0))
    network['12'] = activ_func(conv_func(graph['11'], 2))
    network['p1'] = pool(network['12'])

    #2
    network['21'] = activ_func(conv_func(graph['p1'], 5))
    network['22'] = activ_func(conv_func(graph['21'], 7))
    network['p2'] = pool(network['22'])

    #3
    network['31'] = activ_func(conv_func(graph['p2'], 10))
    network['32'] = activ_func(conv_func(graph['31'], 12))
    network['33'] = activ_func(conv_func(graph['32'], 14))
    network['34'] = activ_func(conv_func(graph['33'], 16))
    network['p3'] = pool(network['34'])
    
    #4
    network['41'] = activ_func(conv_func(graph['p3'], 19))
    network['42'] = activ_func(conv_func(graph['41'], 21))
    network['43'] = activ_func(conv_func(graph['42'], 23))
    network['44'] = activ_func(conv_func(graph['43'], 25))
    network['p4'] = pool(network['44'])
    
    #5
    network['51'] = activ_func(conv_func(graph['p4'], 19))
    network['52'] = activ_func(conv_func(graph['51'], 21))
    network['53'] = activ_func(conv_func(graph['52'], 23))
    network['54'] = activ_func(conv_func(graph['53'], 25))
    network['p5'] = pool(network['54'])
    
    return network







