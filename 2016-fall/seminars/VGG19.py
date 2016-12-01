import numpy as np
import tensorflow as tf
import skimage.transform

MEAN_VALUES = np.array([103.939, 116.779, 123.68]).reshape((3,1,1))

def process_image(im):
    result_w = 224
    h, w, _ = im.shape
    if h < w:
        im = skimage.transform.resize(im, (result_w, w*result_w/h), preserve_range=True)
    else:
        im = skimage.transform.resize(im, (h*result_w/w, result_w), preserve_range=True)
    # Central crop
    h, w, _ = im.shape
    im = im[h//2-result_w//2:h//2+result_w//2, w//2-result_w//2:w//2+result_w//2, :]
    # Shuffle axes
    im = np.swapaxes(np.swapaxes(im, 1, 2), 0, 1)
    # Convert RGB to BGR
    im = im[::-1, :, :]
    im = im - MEAN_VALUES
    return im[np.newaxis, :, :, :]

def max_pool(input_tensor, name):
    with tf.variable_scope(name):
        pool = tf.nn.max_pool(input_tensor, 
                              ksize=[1, 1, 2, 2], 
                              strides=[1, 1, 2, 2], 
                              padding='SAME', 
                              name='pooling',
                              data_format='NCHW')
    return pool

def conv2d_layer(input_tensor, output_channels, name):
    with tf.variable_scope(name):
        kernel = tf.get_variable('kernel', 
                                 [3, 3, input_tensor.get_shape()[1], output_channels],
                                 initializer=tf.truncated_normal_initializer(stddev=1e-2, seed=322), 
                                 dtype=tf.float32)
        conv = tf.nn.conv2d(input_tensor, 
                            kernel, 
                            [1,1,1,1],
                            padding='SAME',
                            data_format='NCHW')
        biases = tf.get_variable('biases', 
                                 [output_channels], 
                                 initializer=tf.constant_initializer(0.0), 
                                 dtype=tf.float32)
        biased = tf.nn.bias_add(conv, biases, data_format='NCHW')
        nonlinear = tf.nn.relu(biased)
    return nonlinear

def dense_layer(input_tensor, num_neurons, name):
    with tf.variable_scope(name) as scope:
        weights = tf.get_variable('weights', 
                                  [input_tensor.get_shape()[1].value, num_neurons],
                                  initializer=tf.truncated_normal_initializer(stddev=1e-2, seed=322),
                                  dtype=tf.float32)
        biases = tf.get_variable('biases', 
                                 [num_neurons], 
                                 initializer=tf.constant_initializer(0.0), 
                                 dtype=tf.float32)
        dense = tf.matmul(input_tensor, weights) + biases
    return dense

def inference(images):
    # block 1
    conv1_1 = conv2d_layer(images, 64, 'conv1_1')
    conv1_2 = conv2d_layer(conv1_1, 64, 'conv1_2')
    pool1 = max_pool(conv1_2, 'pool1')
    # block 2
    conv2_1 = conv2d_layer(pool1, 128, 'conv2_1')
    conv2_2 = conv2d_layer(conv2_1, 128, 'conv2_2')
    pool2 = max_pool(conv2_2, 'pool2')
    # block 3
    conv3_1 = conv2d_layer(pool2, 256, 'conv3_1')
    conv3_2 = conv2d_layer(conv3_1, 256, 'conv3_2')
    conv3_3 = conv2d_layer(conv3_2, 256, 'conv3_3')
    conv3_4 = conv2d_layer(conv3_3, 256, 'conv3_4')
    pool3 = max_pool(conv3_4, 'pool3')
    # block 4
    conv4_1 = conv2d_layer(pool3, 512, 'conv4_1')
    conv4_2 = conv2d_layer(conv4_1, 512, 'conv4_2')
    conv4_3 = conv2d_layer(conv4_2, 512, 'conv4_3')
    conv4_4 = conv2d_layer(conv4_3, 512, 'conv4_4')
    pool4 = max_pool(conv4_4, 'pool4')
    # block 5
    conv5_1 = conv2d_layer(pool4, 512, 'conv5_1')
    conv5_2 = conv2d_layer(conv5_1, 512, 'conv5_2')
    conv5_3 = conv2d_layer(conv5_2, 512, 'conv5_3')
    conv5_4 = conv2d_layer(conv5_3, 512, 'conv5_4')
    pool5 = max_pool(conv5_4, 'pool5')
    # top
    current_channels = pool5.get_shape()[1].value 
    current_height = pool5.get_shape()[2].value 
    current_width = pool5.get_shape()[3].value
    reshaped = tf.reshape(pool5, [-1, current_channels*current_height*current_width])
    dense6 = tf.nn.relu(dense_layer(reshaped, 4096, 'dense6'))
    dropout6 = tf.nn.dropout(dense6, 0.5)
    dense7 = tf.nn.relu(dense_layer(dropout6, 4096, 'dense7'))
    dropout7 = tf.nn.dropout(dense7, 0.5)
    dense8 = tf.nn.softmax(dense_layer(dropout7, 1000, 'dense8'))
    return dense8

def dense6(images):
    # block 1
    conv1_1 = conv2d_layer(images, 64, 'conv1_1')
    conv1_2 = conv2d_layer(conv1_1, 64, 'conv1_2')
    pool1 = max_pool(conv1_2, 'pool1')
    # block 2
    conv2_1 = conv2d_layer(pool1, 128, 'conv2_1')
    conv2_2 = conv2d_layer(conv2_1, 128, 'conv2_2')
    pool2 = max_pool(conv2_2, 'pool2')
    # block 3
    conv3_1 = conv2d_layer(pool2, 256, 'conv3_1')
    conv3_2 = conv2d_layer(conv3_1, 256, 'conv3_2')
    conv3_3 = conv2d_layer(conv3_2, 256, 'conv3_3')
    conv3_4 = conv2d_layer(conv3_3, 256, 'conv3_4')
    pool3 = max_pool(conv3_4, 'pool3')
    # block 4
    conv4_1 = conv2d_layer(pool3, 512, 'conv4_1')
    conv4_2 = conv2d_layer(conv4_1, 512, 'conv4_2')
    conv4_3 = conv2d_layer(conv4_2, 512, 'conv4_3')
    conv4_4 = conv2d_layer(conv4_3, 512, 'conv4_4')
    pool4 = max_pool(conv4_4, 'pool4')
    # block 5
    conv5_1 = conv2d_layer(pool4, 512, 'conv5_1')
    conv5_2 = conv2d_layer(conv5_1, 512, 'conv5_2')
    conv5_3 = conv2d_layer(conv5_2, 512, 'conv5_3')
    conv5_4 = conv2d_layer(conv5_3, 512, 'conv5_4')
    pool5 = max_pool(conv5_4, 'pool5')
    # top
    current_channels = pool5.get_shape()[1].value 
    current_height = pool5.get_shape()[2].value 
    current_width = pool5.get_shape()[3].value
    reshaped = tf.reshape(pool5, [-1, current_channels*current_height*current_width])
    dense6 = tf.nn.relu(dense_layer(reshaped, 4096, 'dense6'))
    return dense6

def assign_weights(trainable_variables, weights_list):
    i = 0
    assign_ops = []
    for v in trainable_variables:
        w = weights_list[i]
        if w.ndim > 2:
            w = np.moveaxis(w, [0,1,2,3], [3, 2, 0, 1])
        assign_ops.append(v.assign(w))
        i += 1
    return assign_ops
