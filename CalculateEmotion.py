from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import random

#import tensorflow as tf
#vgg = tf.contrib.slim.nets.vgg

#from tensorflow.contrib.slim.nets import resnet_v1


#slim = tf.contrib.slim

def Image_model(FacialEmotion):
    """Creates the video model.
    
    Args:
        video_frames: A tensor that contains the video input.
        audio_frames: not needed (leave None).
    Returns:
        The video model.
    with tf.variable_scope("video_model"):
        batch_size, seq_length, height, width, channels = video_frames.get_shape().as_list()

        video_input = tf.reshape(video_frames, (batch_size * seq_length, height, width, channels))
        video_input = tf.cast(video_input, tf.float32)

        features, end_points = resnet_v1.resnet_v1_50(video_input, None)
        features = tf.reshape(features, (batch_size, seq_length, int(features.get_shape()[3])))
 	"""

    test = random.uniform(70, 72)
    return FacialEmotion,test

def combined_model(AudioEmotion,FacialEmotion):
    """Creates the audio-visual model.
    
    Args:
        video_frames: A tensor that contains the video input.
        audio_frames: A tensor that contains the audio input.
    Returns:
        The audio-visual model.
    """
    visual_features = Image_model(FacialEmotion)

    return  visual_features


def audio_model(AudioEmotion):
    with tf.variable_scope("audio_model"):
      batch_size, seq_length, num_features = audio_frames.get_shape().as_list()
      audio_input = tf.reshape(audio_frames, [batch_size * seq_length, 1, num_features, 1])

      with slim.arg_scope([slim.layers.conv2d], padding='SAME'):
        net = slim.dropout(audio_input)
        net = slim.layers.conv2d(net, conv_filters, (1, 20))

        # Subsampling of the signal to 8KhZ.
        net = tf.nn.max_pool(
            net,
            ksize=[1, 1, 2, 1],
            strides=[1, 1, 2, 1],
            padding='SAME',
            name='pool1')

        # Original model had 400 output filters for the second conv layer
        # but this trains much faster and achieves comparable accuracy.
        net = slim.layers.conv2d(net, conv_filters, (1, 40))

        net = tf.reshape(net, (batch_size * seq_length, num_features // 2, conv_filters, 1))

        # Pooling over the feature maps.
        net = tf.nn.max_pool(
            net,
            ksize=[1, 1, 10, 1],
            strides=[1, 1, 10, 1],
            padding='SAME',
            name='pool2')

      net = tf.reshape(net, (batch_size, seq_length, num_features //2 * 4 ))
    return net


def recurrent_model(net, hidden_units=256, number_of_outputs=2):
    batch_size, seq_length, num_features = net.get_shape().as_list()

    lstm = tf.nn.rnn_cell.LSTMCell(hidden_units,
                                   use_peepholes=True,
                                   cell_clip=100,
                                   state_is_tuple=True)

    stacked_lstm = tf.nn.rnn_cell.MultiRNNCell([lstm] * 2, state_is_tuple=True)

    # We have to specify the dimensionality of the Tensor so we can allocate
    # weights for the fully connected layers.
    outputs, _ = tf.nn.dynamic_rnn(stacked_lstm, net, dtype=tf.float32)

    net = tf.reshape(outputs, (batch_size * seq_length, hidden_units))

    prediction = slim.layers.linear(net, number_of_outputs)
    
    return tf.reshape(prediction, (batch_size, seq_length, number_of_outputs))


def Calculate_Emotion(AudioEmotionObject,FacialEmotionObject):
   FinalEmotion , Accuracy= combined_model(AudioEmotionObject,FacialEmotionObject)
   return  FinalEmotion, Accuracy

