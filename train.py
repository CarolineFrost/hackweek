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
style_weights = [('11', 0.5), ('21', 1.0), ('31', 1.5),('41', 3.0),('51', 4.0)]

#Content loss
def L_content(sess, model):
    #i is content layer
    i = 2
    #x is generated image
    #F is feature representation of x in given layer Li
    F = sess.run(model[i])
    #p is original image
    #P is feature representation of p in given layer Li
    P = model[i]
    loss = 0.5 * tf.reduce_sum(tf.pow(F - P, 2))
    return loss

#Style loss
def L_style(sess, model, style_weights):
    
    def layer_loss(out1, out2):
        G = np.reshape(out1, (out1.shape[2] * out1.shape[1], out1.shape[3]))
        A = np.reshape(out2, (out2.shape[2] * out2.shape[1], out2.shape[3]))
        squared_sumGA = tf.reduce_sum(tf.pow(G - A, 2))
        return (1.0 / (4 * N**2 * M**2)) * squared_sumGA
    
    total_loss = 0.0
    for layer, weight in style_weights:
       temp_loss = layer_loss(sess.run(model[layer]), model[layer])
        total_loss += weight * temp_loss

    return total_loss

if __name__ == '__main__':
    session = tf.session()

