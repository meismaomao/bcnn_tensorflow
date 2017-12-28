from dann_utils import *
import tensorflow as tf

class BcnnModel(object):
    """Simple MNIST Bilinear CNN Model -- Two CNN Network, Two CNN Data Stream!!!!!!"""
    def __init__(self, batch_size, pixel_mean):
        self.pixel_mean = pixel_mean
        self.batch_size = batch_size
        self._build_model()

    def _build_model(self):
        self.X = tf.placeholder(tf.uint8, [None, 28, 28, 3])
        self.y = tf.placeholder(tf.float32, [None, 10])


        X_input = (tf.cast(self.X, tf.float32) - self.pixel_mean) / 255.

        # CNN model for feature extraction
        with tf.variable_scope('feature_extractor1'):
            W_conv0 = weight_variable([5, 5, 3, 32])
            b_conv0 = bias_variable([32])
            h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
            h_pool0 = max_pool_2x2(h_conv0)

            W_conv1 = weight_variable([5, 5, 32, 48])
            b_conv1 = bias_variable([48])
            h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
            self.h_pool1 = max_pool_2x2(h_conv1)

        with tf.variable_scope('feature_extractor2'):
        	W_conv0 = weight_variable([5, 5, 3, 32])
        	b_conv0 = bias_variable([32])
        	h_conv0 = tf.nn.relu(conv2d(X_input, W_conv0) + b_conv0)
        	h_pool0 = max_pool_2x2(h_conv0)

        	W_conv1 = weight_variable([5, 5, 32, 48])
        	b_conv1 = bias_variable([48])
        	h_conv1 = tf.nn.relu(conv2d(h_pool0, W_conv1) + b_conv1)
        	self.h_pool2 = max_pool_2x2(h_conv1)

        # Bilinear vector for class prediction
        with tf.variable_scope('label_predictor'):

            conv0 = tf.transpose(self.h_pool1, perm=[0, 3, 1, 2])
            conv0_1 = tf.transpose(self.h_pool2, perm=[0, 3, 1, 2])

            conv0 = tf.reshape(conv0, [-1, 48, 14 * 14])
            conv0_1 = tf.reshape(conv0_1, [-1, 48, 14 * 14])

            conv0_1_T = tf.transpose(conv0_1, [0, 2, 1])

            phi_I = tf.matmul(conv0, conv0_1_T)

            phi_I = tf.rehsape(phi_I, [-1, 48 * 48])

            phi_I = tf.divide(phi_I, 196.0)

            y_sqrt = tf.multiply(tf.sign(phi_I), tf.sqrt(tf.abs(phi_I) + 1e-12))

            z_l2 = tf.nn.l2_normalize(y_sqrt, dim=1)

            W_fc0 = weight_variable([48 * 48, 100])
            b_fc0 = bias_variable([100])
            h_fc0 = tf.nn.relu(tf.matmul(z_l2, W_fc0) + b_fc0)

            W_fc1 = weight_variable([100, 100])
            b_fc1 = bias_variable([100])
            self.visual_feature = tf.matmul(h_fc0, W_fc1) + b_fc1
            h_fc1 = tf.nn.relu(tf.matmul(h_fc0, W_fc1) + b_fc1)

            W_fc2 = weight_variable([100, 10])
            b_fc2 = bias_variable([10])
            logits = tf.matmul(h_fc1, W_fc2) + b_fc2

            self.pred = tf.nn.softmax(logits)
            # print(logits, self.classify_labels)
            self.pred_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.classify_labels, logits=logits)

