'''
# Time   : 2020/10/21 16:55
# Author : junchaoli
# File   : model.py
'''

import tensorflow as tf
import tensorflow.keras.backend as K

class Logistic_layer(tf.keras.layers.Layer):
    def __init__(self, k, w_reg):
        super().__init__()
        self.k = k
        self.w_reg = w_reg

    def build(self, input_shape):
        self.w0 = self.add_weight(name='w0', shape=(1,),
                                  initializer=tf.zeros_initializer(),
                                  trainable=True,)
        self.w = self.add_weight(name='w', shape=(input_shape[-1], 1),
                                 initializer=tf.random_normal_initializer(),
                                 trainable=True,
                                 regularizer=tf.keras.regularizers.l2(self.w_reg))

    def call(self, inputs, **kwargs):
        if K.ndim(inputs) != 2:
            raise ValueError("Unexpected inputs dimensions %d, expect to be 2 dimensions" % (K.ndim(inputs)))

        linear_part = tf.matmul(inputs, self.w) + self.w0   #shape:(batchsize, 1)
        output = linear_part
        return tf.nn.sigmoid(output)

class Logistic(tf.keras.Model):
    def __init__(self, k, w_reg=1e-4):
        super().__init__()
        self.logistic = Logistic_layer(k, w_reg)

    def call(self, inputs, training=None, mask=None):
        output = self.logistic(inputs)
        return output
