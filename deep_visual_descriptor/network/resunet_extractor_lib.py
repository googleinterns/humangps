# Copyright 2021 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Library defining the architecture of residual U-Net."""

from typing import Tuple, List, Sequence, Union

import tensorflow as tf
import tensorflow_addons as tfa


class ResidualBlock(tf.keras.layers.Layer):
  """A residual block for residual U-Net."""

  def __init__(self,
               num_filters: int,
               kernel_size: int = 3,
               strides: int = 1,
               padding: str = 'same',
               norm_fn: str = 'instance',
               **kwargs):
    """Initializes the residual block.

    Args:
      num_filters: An integer specifying the number of filters in convolution.
      kernel_size: An integer specifying the number of kernel size of the
        convolutions in residual block.
      strides: An integer specifying the strides of the convolution along the
        height and width.
      padding: A string specifying the convolution padding style.
      norm_fn: A string specifying the type of normalization layer.
      **kwargs: Keyworded arguments that are forwarded by the residual block.

    Raises:
      Exception: if the normalization type of 'norm_fn' is not supported.
    """
    super(ResidualBlock, self).__init__(**kwargs)

    self.conv1 = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        activation=None)
    self.conv2 = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=kernel_size,
        strides=1,
        padding=padding,
        activation=None)
    self.skip_conv = tf.keras.layers.Conv2D(
        filters=num_filters,
        kernel_size=1,
        strides=strides,
        padding=padding,
        activation=None)

    beta_initializer = 'zeros'
    gamma_initializer = 'ones'
    if norm_fn == 'instance':
      self.norm1 = tfa.layers.InstanceNormalization(
          axis=-1,
          epsilon=1e-5,
          center=False,
          scale=False,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.norm2 = tfa.layers.InstanceNormalization(
          axis=-1,
          epsilon=1e-5,
          center=False,
          scale=False,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.skip_norm = tfa.layers.InstanceNormalization(
          axis=-1,
          epsilon=1e-5,
          center=False,
          scale=False,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
    elif norm_fn == 'batch':
      self.norm1 = tf.keras.layers.BatchNormalization(
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.norm2 = tf.keras.layers.BatchNormalization(
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
      self.skip_norm = tf.keras.layers.BatchNormalization(
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
    else:
      raise Exception('norm_fn %s not implemented' % norm_fn)

    self.relu = tf.keras.layers.ReLU()

  def call(self, inputs: tf.Tensor, training: bool = True) -> tf.Tensor:
    """Run the forward-pass operations of the residual block.

    Args:
      inputs: A Tensor of size B x H x W x C where B is batch size, H x W is the
        image size, and C is the number of channels.
      training: A boolean that defines whether the call should be excuted as a
        training or an inference call.

    Returns:
      outputs: The features of size B x (H+1)//S x (W+1)//S x K, where K
        is the number of filters, S is the strides.
    """
    outputs = self.relu(self.conv1(self.norm1(inputs, training=training)))
    outputs = self.relu(self.conv2(self.norm2(outputs, training=training)))
    skip_outputs = self.skip_norm(self.skip_conv(inputs), training=training)

    return outputs + skip_outputs


class ResUNet(tf.keras.Model):
  """Build a Residual UNet."""

  def __init__(self,
               filters_sequence: List[int] = (16, 32, 64, 96, 128, 128, 196),
               output_channel: int = 16,
               num_resblock_per_level: int = 2,
               norm_fn: str = 'instance',
               return_feature_pyramid: bool = False,
               name: str = 'ResUNet',
               **kwargs):
    """Initializes the residual U-Net.

    Args:
      filters_sequence: A sequence of integers specifying the number of
        convolutional filters in each residual block.
      output_channel: An integer specifying the number of channels in the final
        decoder output.
      num_resblock_per_level: A sequence of integers specifying the number of
        residual block per level.
      norm_fn: A string specifying the type of normalization layer.
      return_feature_pyramid: If true, the call function will also return the
        whole decoder pyramid.
      name: A string specifying a name for the model.
      **kwargs: Keyworded arguments that are forwarded by the ResUnet.

    Raises:
      Exception: if the normalization type of 'norm_fn' is not supported.
    """
    super(ResUNet, self).__init__(name=name, **kwargs)

    self._filters_sequence = filters_sequence
    self._return_feature_pyramid = return_feature_pyramid
    beta_initializer = 'zeros'
    gamma_initializer = 'ones'

    # Initialize layers for encoder.
    self.stem_conv1 = tf.keras.layers.Conv2D(
        filters=self._filters_sequence[0],
        kernel_size=3,
        strides=1,
        padding='same',
        activation=None)
    self.stem_conv2 = self._make_conv_block(self._filters_sequence[0])
    self.stem_skip_conv = tf.keras.layers.Conv2D(
        filters=self._filters_sequence[0],
        kernel_size=1,
        strides=1,
        padding='same',
        activation=None)
    if norm_fn == 'instance':
      self.stem_skip_norm = tfa.layers.InstanceNormalization(
          axis=-1,
          epsilon=1e-5,
          center=False,
          scale=False,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
    elif norm_fn == 'batch':
      self.stem_skip_norm = tf.keras.layers.BatchNormalization(
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
    else:
      raise Exception('norm_fn %s not implemented' % norm_fn)

    self.encode_layers = [None] * (len(self._filters_sequence) - 1)
    for idx, _ in enumerate(self.encode_layers):
      layers = [
          ResidualBlock(num_filters=self._filters_sequence[idx + 1], strides=1),
          ResidualBlock(num_filters=self._filters_sequence[idx + 1], strides=2)
      ]
      self.encode_layers[idx] = tf.keras.Sequential(layers=layers)

    self.decode_layers = [None] * (len(self._filters_sequence) - 1)
    for idx, _ in enumerate(self.decode_layers):
      layers = [
          ResidualBlock(num_filters=self._filters_sequence[::-1][idx]),
          ResidualBlock(num_filters=self._filters_sequence[::-1][idx])
      ]
      self.decode_layers[idx] = tf.keras.Sequential(layers=layers)

    # Initialize layers for bridge.
    self.bridge_conv1 = self._make_conv_block(self._filters_sequence[-1])
    self.bridge_conv2 = self._make_conv_block(self._filters_sequence[-1])

    # Initialize output layers.
    self.decode_output_layer = [None] * (len(self._filters_sequence) - 1)
    for idx, _ in enumerate(self.decode_output_layer):
      self.decode_output_layer[idx] = tf.keras.layers.Conv2D(
          output_channel,
          kernel_size=1,
          strides=1,
          padding='same',
          activation=None)

    self.upsample_layer = tf.keras.layers.UpSampling2D(size=2, name='upsample')
    self.concatenation_layer = tf.keras.layers.Concatenate(
        axis=-1, name='concatenate')

  def _make_conv_block(self,
                       filters=16,
                       kernel_size=3,
                       strides=1,
                       norm_fn: str = 'instance',
                       padding='same') -> tf.keras.layers.Layer:
    """Build a convolution block.

    Args:
      filters: A integer specifying the number of filters in convolution.
      kernel_size: A integer specifying the number of kernel size of the
        convolutions in residual block.
      strides: A integer specifying the strides of the convolution along the
        height and width.
      norm_fn: A string specifying the type of normalization layer.
      padding: A string specifying the convolution padding style.

    Returns:
      Convolution block: A layer combining the normalization layer, ReLU and
      convolution.

    Raises:
      Exception: if the normalization type of 'norm_fn' is not supported.
    """
    beta_initializer = 'zeros'
    gamma_initializer = 'ones'
    if norm_fn == 'instance':
      norm = tfa.layers.InstanceNormalization(
          axis=-1,
          epsilon=1e-5,
          center=False,
          scale=False,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
    elif norm_fn == 'batch':
      norm = tf.keras.layers.BatchNormalization(
          epsilon=1e-5,
          beta_initializer=beta_initializer,
          gamma_initializer=gamma_initializer)
    else:
      raise Exception('norm_fn %s not implemented' % norm_fn)

    relu = tf.keras.layers.ReLU()
    conv = tf.keras.layers.Conv2D(
        filters=filters,
        kernel_size=3,
        strides=strides,
        padding=padding,
        activation=None)
    layers = (norm, relu, conv)
    return tf.keras.Sequential(layers=layers)

  def call(
      self,
      inputs: tf.Tensor,
      training: bool = True
  ) -> Union[tf.Tensor, Tuple[tf.Tensor, Sequence[tf.Tensor]]]:
    """Run the forward-pass operations of the residual U-Net.

    Args:
      inputs: A Tensor of size B x H x W x C where B is batch size, H x W is the
        image size, and C is the number of channels.
      training: A boolean that defines whether the call should be excuted as a
        training or an inference call.

    Returns:
      outputs: The features of size B x H x W x K, where K is the
      output_channel. Note that in case return_feature_pyramid is set to True,
      the output will be a tuple as follow (standard output, feature_pyramid).
    """
    # Build encoder.
    outputs = self.stem_conv1(inputs)
    outputs = self.stem_conv2(outputs)
    skip_outputs = self.stem_skip_conv(inputs)
    skip_outputs = self.stem_skip_norm(skip_outputs)
    encode_feature = outputs + skip_outputs
    encode_features = []
    encode_features.append(encode_feature)
    for layer in self.encode_layers:
      encode_feature = layer(encode_feature)
      encode_features.append(encode_feature)

    # Build bridge.
    bridge_feature = self.bridge_conv1(encode_feature)
    bridge_feature = self.bridge_conv2(bridge_feature)
    decode_feature = bridge_feature
    decode_features = []

    # Build decoder.
    for idx, layer in enumerate(self.decode_layers):
      upsample = self.upsample_layer(decode_feature)
      concatenate = self.concatenation_layer(
          [upsample, encode_features[::-1][idx + 1]])
      decode_feature = layer(concatenate)
      decode_features.append(decode_feature)

    # Build output feature pyramid.
    self.output_feature_pyramid = []
    for idx, layer in enumerate(self.decode_output_layer):
      output_feature = self.decode_output_layer[idx](decode_features[idx])
      self.output_feature_pyramid.append(output_feature)

    final_output = self.output_feature_pyramid[-1]
    if self._return_feature_pyramid:
      final_output = (final_output, self.output_feature_pyramid)
    return final_output
