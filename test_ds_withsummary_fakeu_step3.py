#!/usr/bin/env python
# -*- coding: utf-8 -*-


# Jidan Zhong 
# 2017- Jan-19

# load in images and train images for segmentation purpose
import io
import tensorflow as tf
import os
from sklearn.utils import shuffle
import numpy as np
import cv2
##
import random
import time
import matplotlib.pyplot as plt
from PIL import Image
##
#################################################################################
# input from user

OVERALLSIZE = int(input('Choose the size of data : ')) # number of training data
TESTSIZE = int(input('Choose the number of data point you want to use as test : '))
#################################################################################
# other code
PATH = os.path.abspath("test.py")[:-8]
IMAGES = [img for img in os.listdir(PATH + '/imageds') if img.endswith('jpg')]  

# shuffle images and choose the number of overall data
def imageMaker():
    # Shuffle the images name
    imagesIn = IMAGES 
    random.seed(4)
    random.shuffle(imagesIn)
    # imagesIn = shuffle(IMAGES)
    # Take on the relative masks name based directly on the images name
    masksIn = [name[:-13]+'_seg'+name[-13:] for name in imagesIn]

    # Crop the list to user given size: training data
    images, masks = imagesIn[:OVERALLSIZE], masksIn[:OVERALLSIZE]

    # Import the image left as grayscale numpy array using cv2.imread
    images_, masks_  = [cv2.imread(PATH + '/imageds/' + img, cv2.IMREAD_GRAYSCALE).astype(np.int) for img in images], \
                              [cv2.imread(PATH + '/labelds/' + msk, cv2.IMREAD_GRAYSCALE).astype(np.int) for msk in masks]

    return images_, masks_
def save_plot():
    """Save a pyplot plot to buffer."""
    # plt.figure()
    # plt.plot(x, y)
    # plt.title("test")
    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    # Add the batch dimension
    image = tf.expand_dims(image, 0)
    # Add image summary
    summary_op = tf.image_summary("plot", image)
    # return buf
    return summary_op

def variable_summaries(var):                ###################################################################ADDDDDED########
    """Attach a lot of summaries to a Tensor (for TensorBoard visualization)."""
    with tf.name_scope('summaries'):
        mean = tf.reduce_mean(var)
        tf.summary.scalar('mean', mean)
    with tf.name_scope('stddev'):
        stddev = tf.sqrt(tf.reduce_mean(tf.square(var - mean)))
        tf.summary.scalar('stddev', stddev)
        tf.summary.scalar('max', tf.reduce_max(var))
        tf.summary.scalar('min', tf.reduce_min(var))
        tf.summary.histogram('histogram', var)

#  define weights and biases, initialization
def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

# convolution and pooling
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME')


def convoer(inputs, shape, flag, layer_name):
    # Create parameters
    with tf.name_scope(layer_name):
        with tf.name_scope('weights1'):
            W = weight_variable(shape)
            variable_summaries(W)
        with tf.name_scope('biases1'):
            b = bias_variable([shape[3]])
            variable_summaries(b)
        temp = shape
        temp[2] = shape[3]

        # Step convolution with inputs
        with tf.name_scope('conv1'):
            preactivate = conv2d(inputs, W) + b
            tf.summary.histogram('pre_activations1', preactivate)
        conv = tf.nn.relu(preactivate, name='activations1')
        tf.summary.histogram('activations1', conv)  

        with tf.name_scope('weights2'):
            Wa = weight_variable(temp)
            variable_summaries(Wa)
        with tf.name_scope('biases2'):
            ba = bias_variable([shape[3]])
            variable_summaries(ba)  

        # Step another convolution with conv to allow further tuning  
        with tf.name_scope('conv2'):
            preactivate = conv2d(conv, Wa) + ba
            tf.summary.histogram('pre_activations2', preactivate)
        conv = tf.nn.relu(preactivate, name='activations2')
        tf.summary.histogram('activations12', conv)
        
        with tf.name_scope('maxpool'):
        # Max pool over 2x2 section (allow model to be rotational, sliding and basic transformations independent, also reduce dimensionality without giving up too much of the informations gained)
            pool = max_pool_2x2(conv)
            tf.summary.histogram('pool', pool)
        # Allow convenience to conv10 and conv6
        if flag: 
            return pool
        elif not flag: 
            return conv

# Create a deconvolution pooling step
def upconvoer(inputs, shape, height, width, layer_name):
    # Create parameters
    
    # Resize the matrix to upper given dimensions
    up = tf.image.resize_images(inputs, [height, width]) 

    with tf.name_scope(layer_name):
        with tf.name_scope('weights1'):
            W = weight_variable(shape)
            variable_summaries(W)
        with tf.name_scope('biases1'):
            b = bias_variable([shape[3]])
            variable_summaries(b)  

       # Step convolution with inputs
        with tf.name_scope('upconv1'):
            preactivate = conv2d(up, W) + b  #############  INPUTS    ###########
            tf.summary.histogram('pre_activations1', preactivate)
            conv = tf.nn.relu(preactivate, name='activations1')
            tf.summary.histogram('activations1', conv)  

        temp = shape
        temp[2] = shape[3]  

        with tf.name_scope('weights2'):
            Wa = weight_variable(temp)
            variable_summaries(Wa)
        with tf.name_scope('biases2'):
            ba = bias_variable([shape[3]])
            variable_summaries(ba)  

        with tf.name_scope('upconv2'):
            preactivate = conv2d(conv, Wa) + ba
            tf.summary.histogram('pre_activations2', preactivate)
            conv = tf.nn.relu(preactivate, name='activations2')
            tf.summary.histogram('activations12', conv)
    return conv



# Split the dataset in test train and normalizing to avoid too big number
# which are hard to handle within TensorFlow
ugod = imageMaker()
images_, masks_ = ugod[0], ugod[1]
# first TESTSIZE cases are testing data; following are trainining data.
X_train, y_train, X_test, y_test = np.asarray(images_[TESTSIZE:])/255., \
                                      np.asarray(masks_[TESTSIZE:])/255, \
                                      np.asarray(images_[:TESTSIZE])/255., \
                                      np.asarray(masks_[:TESTSIZE])/255

Nsize = 96  # image resolution

with tf.Graph().as_default():
    with tf.name_scope('input'):

        # Create 3D placeholder to contain the inputs [batch, height+grayscale, width+grascale] (tensor will further be change to 4D [batch, height, width, grayscale])
        x = tf.placeholder(tf.float32, shape=[None, Nsize, Nsize], name='x-input')
        # Create a 4D placeholder to contain label inputs [batch, height, width, grascale]
        y_ = tf.placeholder(tf.float32, shape=[None, Nsize, Nsize,1], name='y-input')

    #########

    with tf.name_scope('input_reshape'):   ###################################################################ADDDDDED########
        inputs = tf.reshape(x,[-1,Nsize,Nsize,1])
        tf.summary.image('input', inputs, 4) # 10: show 10 images

        # Construct the Unet
        #with tf.device('/cpu:0'):

    with tf.name_scope('Model'):

        pool1 = convoer(inputs, [3,3,1,12], True,'layer1')
        pool2 = convoer(pool1, [3,3,12,24], True,'layer2')
        pool3 = convoer(pool2, [3,3,24,48], True,'layer3')
        pool4 = convoer(pool3, [3,3,48,96], True,'layer4')
        conv5 = convoer(pool4, [3,3,96,192], False,'layer5')

        conv6 = upconvoer(conv5, [3,3,192,96], 12, 12,'layer6')
        conv7 = upconvoer(conv6, [3,3,96,48], 24, 24,'layer7')
        conv8 = upconvoer(conv7, [3,3,48,24], 48, 48,'layer8')
        # deconvolution leads to same size image
        conv9 = upconvoer(conv8, [3,3,24,12], Nsize, Nsize,'layer9')

        with tf.name_scope('layer10'):
            with tf.name_scope('weights'):
                W10 = weight_variable([1,1,12,2])
                variable_summaries(W10)
            with tf.name_scope('biases'):
                b10 = bias_variable([2])
                variable_summaries(b10)
            with tf.name_scope('conv'):
                preactivate = conv2d(conv9, W10) + b10 
                tf.summary.histogram('pre_activations', preactivate)
            y = tf.nn.sigmoid(preactivate, name='activation')
            tf.summary.histogram('activations', y)


    #Train and Evaluate the Model

    #########label for 2 channels

    class_labels_tensor = tf.equal(y_, 1)
    background_labels_tensor = tf.not_equal(y_, 1)
    # Convert the boolean values into floats -- so that computations in cross-entropy loss is correct
    bit_mask_class = tf.to_float(class_labels_tensor)
    bit_mask_background = tf.to_float(background_labels_tensor)

    combined_mask = tf.concat(concat_dim=3, values=[bit_mask_class, bit_mask_background])
    # Lets reshape our input so that it becomes suitable for 
    # tf.softmax_cross_entropy_with_logits with [batch_size, num_classes]
    # flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, 2))
    flat_labels = tf.reshape(tensor=combined_mask, shape=(-1, Nsize*Nsize, 2))

    flat_logits = tf.reshape(tensor=y, shape=(-1, Nsize*Nsize, 2))

    with tf.name_scope('cross_entropy'):

        label_val = flat_labels[:,:,0]
        logit_val = flat_logits[:,:,0]
        intersection = tf.multiply(label_val, logit_val)
        union = tf.subtract( tf.add(label_val, logit_val), intersection)
        dice = tf.div(tf.reduce_sum(intersection, axis=1), tf.reduce_sum(union, axis=1))


        with tf.name_scope('total'):
            # cross_entropy_sum = tf.reduce_mean(cross_entropy)  # sum
            cross_entropy_sum = tf.reduce_mean(-dice)
            tf.summary.scalar('cross_entropy', cross_entropy_sum)      

    with tf.name_scope('train'):
        train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_sum)

    ##################################################################################

    # Merge all the summaries and write them out to /tmp/mnist_logs (by default)
    merged = tf.summary.merge_all()

    saver = tf.train.Saver()
    start = time.time()

    with tf.Session(config=tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)) as sess: # the first one means if the device doesnt exist, it can automatically appoint an existing device; 2nd means it will show the log infor for parameters and operations are on which device
        train_writer = tf.summary.FileWriter('/home/jidan/test/train', sess.graph)
        test_writer = tf.summary.FileWriter('/home/jidan/test/test')
        
        tf.global_variables_initializer().run()

        N1 = 100
        EPOCH_LIMIT = 51#00
        for E in range(EPOCH_LIMIT):
            for i in range(len(X_train)-N1):                     
                if i%N1 == 0:
                    # resize the y_train to meet its placeholder dimension of 4D
                    batch = y_train[i:i+N1]
                    joe = np.resize(batch, [len(batch), Nsize, Nsize, 1])
                    joe = joe.astype(np.float32)                    
                    loss, pred, summary,_ = sess.run([cross_entropy_sum, y, merged,train_step], feed_dict={x: X_train[i:i+N1], y_: joe})
                    # train_writer.add_summary(summary, i)
                    print 'epoch %d step %d, loss %g' %(E, i,loss)
            i = len(X_train) - N1        
            batch = y_train[i:i+N1]
            joe = np.resize(batch, [len(batch), Nsize, Nsize, 1])
            joe = joe.astype(np.float32)
            
            loss, pred, summary,_ = sess.run([cross_entropy_sum, y, merged,train_step], feed_dict={x: X_train[i:i+N1], y_: joe})

            train_writer.add_summary(summary, E)
            print 'epoch %d step %d, loss %g' %(E, i,loss)      

            # plot into graph
            batch = y_train[0:N1]
            joe = np.resize(batch, [len(batch), Nsize, Nsize, 1])
            joe = joe.astype(np.float32)  
            loss, pred = sess.run([cross_entropy_sum, y], feed_dict={x: X_train[0:N1], y_: joe})

            # plt.ion()
            plt.figure()
            plt.subplot(321),plt.imshow(X_train[0,:,:],cmap = 'gray')
            plt.title('Training Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(323),plt.imshow(joe[0,:,:,0],cmap = 'gray')
            plt.title('Mask Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(325),plt.imshow(pred[0,:,:,0],cmap = 'gray')
            plt.title('Predicted Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(322),plt.imshow(X_train[1,:,:],cmap = 'gray')
            plt.title('Training Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(324),plt.imshow(joe[1,:,:,0],cmap = 'gray')
            plt.title('Mask Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(326),plt.imshow(pred[1,:,:,0],cmap = 'gray')
            plt.title('Predicted Image'), plt.xticks([]), plt.yticks([])
            # plt.show()
            # plt.draw()
            # plt.pause(0.001)
            summary_op = save_plot()
            summary1 = sess.run(summary_op)
            train_writer.add_summary(summary1)

            batch = y_test
            joet = np.resize(batch, [len(batch), Nsize, Nsize, 1])
            joet = joet.astype(np.float32)
            loss, pred, summary = sess.run([cross_entropy_sum, y, merged], feed_dict={x: X_test, y_: joet})
            test_writer.add_summary(summary, E)
            
            # plt.ion()
            plt.figure()
            plt.subplot(321),plt.imshow(X_test[0,:,:],cmap = 'gray')
            plt.title('Testing Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(323),plt.imshow(joet[0,:,:,0],cmap = 'gray')
            plt.title('Mask Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(325),plt.imshow(pred[0,:,:,0],cmap = 'gray')
            plt.title('Predicted Image'), plt.xticks([]), plt.yticks([])

            plt.subplot(322),plt.imshow(X_test[1,:,:],cmap = 'gray')
            plt.title('Testing Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(324),plt.imshow(joet[1,:,:,0],cmap = 'gray')
            plt.title('Mask Image'), plt.xticks([]), plt.yticks([])
            plt.subplot(326),plt.imshow(pred[1,:,:,0],cmap = 'gray')
            plt.title('Predicted Image'), plt.xticks([]), plt.yticks([])
            # plt.show()        
            # plt.draw()
            # plt.pause(0.001)
            summary_op = save_plot()
            summary1 = sess.run(summary_op)
            test_writer.add_summary(summary1)
            end = time.time()-start
            print 'total time spent running %f' %(end)

            if E % 10 == 0:
                saver.save(sess, '/media/truecrypt1/Research/model', global_step=E)

                end = time.time()-start
                print 'total time spent running %f' %(end)

        end = time.time()-start
        print 'total time spent running %f' %(end)
