"""
   Written by Pengfei Sun. 
"""

import tensorflow as tf

class DFiltering(object):
      """ Dilated filtering system for feature extraction. 
          Basically it incorporates the dilated cnn and the memorizing cell of LSTM to realize information transfer
          in temporal scale and frequecy scale. 
      """
      def __init__(is_training, input_tensor, num_blocks, dim):
          self.is_training=is_training
          self.input_tensor=input_tensor
          self.num_blocks=num_blocks
          self.dim=dim
      
      def conv1d_layer(self, input_tensor, size=1, dim=128,
                        is_training=True,  bias=False,
                 activation="tanh", scope=None):
          dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
          with tf.variable_scope(scope):
               shape  = input_tensor.get_shape().as_list()
               kernel_1 = tf.get_variable('kernel_1',
                              (size, shape[-1], dim), # input channel
                              initializer=tf.contrib.layers.xavier_initializer(),
                              dtype=tf.float32)
               if bias:
                    b_1 = variable_on_cpu('b_1', [dim], tf.constant_initializer(0.0))

               out  = tf.nn.conv1d(input_tensor, kernel_1, stride=1, padding="SAME") + \
                                                              (b_1 if bias else 0)
               if not bias:
                  out  = batch_norm_wrapper(out, is_training)

               out = activation_wrapper(out, activation)
         
               return out
               
      def aconv1d_layer(input_tensor,size=7, rate=2, is_training=True, bias=False,
                        activation="tanh", scope=None):
          dtype = tf.float16 if FLAGS.use_fp16 else tf.float32
          with tf.variable_scope(scope):
               shape = input_tensor.get_shape().as_list()
               kernel_2 = tf.get_variable('kernel_2',
                               (1, size, shape[-1], shape[-1]),
                               initializer=tf.contrib.layers.xavier_initializer(),
                               dtype=tf.float32)

               if bias:
                  b_2 = variable_on_cpu('b_2', [shape[-1]], tf.constant_initializer(0.0))

               out  = tf.nn.atrous_conv2d(tf.expand_dims(input_tensor, 1), kernel_2,  \
                                    rate=rate, padding='SAME') + (b_2 if bias else 0)
               out  = tf.squeeze(out, 1)
               if not bias:
                  out = batch_norm_wrapper(out, is_training)
               out  = activation_wrapper(out, activation)

               return out
               
      def residual_block(self, input_tensor, output_tensor, size, rate,
                          block, dim, is_training=True):
          with tf.variable_scope("block_%d_%d" % (block, rate)):
      #         conv_filter = aconv1d_layer(input_tensor, size=size,
      #                                     rate=rate, is_training=is_training,
      #                                     activation="tanh", scope="conv_filter")

               conv_filter = aconv1d_layer(input_tensor, size=size,
                                     rate=rate, is_training=is_training,
                                     activation="tanh", scope="conv_filter")
               if output_tensor ==0:
                  conv_gate   = aconv1d_layer(input_tensor, size=size,
                                     rate=rate, is_training=is_training,
                                     activation="sigmoid", scope="conv_gate")
               else:
                  conv_gate   = aconv1d_layer(output_tensor, size=size,
                                     rate=rate, is_training=is_training,
                                     activation="sigmoid", scope="conv_gate")
               output_ = conv_filter * conv_gate
               
               output_ = conv1d_layer(output_, size=1, dim = dim, activation=None,
                                 is_training=is_training, scope= "conv_out")
               
               return output_ + input_tensor, output_tensor + output_
               
         def deliated_f(self):
             with tf.variable_scope("embedding_layers"):
                   layer_input = self.conv1d_layer(self.input_tensor, dim=nfilters,
                                     is_training=self.is_training, scope="conv_in")
                   layer_output= 0
                   for i in range(self.num_blocks):
                        for i in [1, 2, 4, 8, 16]:  # dilated scope
                             layer_input, layer_output = residual_block(layer_input, layer_output, size=11, \
                                             rate=r, block=i, dim=FLAGS.rnn_size, \
                                             is_training=is_training)
                   return layer_output 
               
Class OctDilating(object):
      """ Octave dilation CNN.
          The model is proposed to overcome the deficit of spectral bias of neural networks. 
          Dilated CNN is applied to obtain different frequecy band information, but limited by 
          neural network are easily over-emphsizing the low frequency components. Octave operation will 
          be applied layer-wise. 
      """
      

               
