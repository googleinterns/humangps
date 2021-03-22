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

"""Tests for google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor.network.resunet_extractor_lib."""
import itertools
from absl.testing import parameterized
import tensorflow as tf

from google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor.network import resunet_extractor_lib

_BATCH_SIZE = (1,)
_IMAGE_HEIGHT = (16,)
_IMAGE_WIDTH = (16,)
_IMAGE_CHANNEL = (1,)

_FILTERS_SEQUENCE = ((2, 4, 4),)
_NUM_FILTERS = (2,)
_OUTPUT_CHANNEL = (2,)
_NORM_FN = ('instance', 'batch')
_RETURN_FEATURE_PYRAMID = (True, False)


def _combine(*tuples):
  return tuple(itertools.product(*tuples))


def _generate_data_batch(batch_size, height, width, channel):
  """Generate a batch of data for testing."""
  data_batch = tf.ones((batch_size, height, width, channel), tf.float32)
  return data_batch


class ResunetExtractorLibTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(*_combine(
      _BATCH_SIZE, _IMAGE_HEIGHT, _IMAGE_WIDTH, _IMAGE_CHANNEL,
      _FILTERS_SEQUENCE, _OUTPUT_CHANNEL, _NORM_FN, _RETURN_FEATURE_PYRAMID))
  def test_resunet_call(self, batch_size, image_height, image_width,
                        image_channel, filters_sequence, output_channel,
                        norm_fn, return_feature_pyramid):
    inputs = _generate_data_batch(batch_size, image_height, image_width,
                                  image_channel)

    resunet_extractor = resunet_extractor_lib.ResUNet(
        filters_sequence=filters_sequence,
        output_channel=output_channel,
        norm_fn=norm_fn,
        return_feature_pyramid=return_feature_pyramid)

    output = resunet_extractor(inputs)

    if not return_feature_pyramid:
      # Validate the shape of the res-Unet output.
      self.assertSequenceEqual(
          output.shape, (batch_size, image_height, image_width, output_channel))
    else:
      prediction, feature_pyramid = output
      self.assertSequenceEqual(
          prediction.shape,
          (batch_size, image_height, image_width, output_channel))
      # Validate the shape of the output feature pyramid.
      height = image_height
      width = image_width
      for decode_output in feature_pyramid[::-1]:
        self.assertSequenceEqual(decode_output.shape,
                                 (batch_size, height, width, output_channel))
        height = height // 2
        width = width // 2

  @parameterized.parameters(*_combine(_BATCH_SIZE, _IMAGE_HEIGHT, _IMAGE_WIDTH,
                                      _IMAGE_CHANNEL, _NUM_FILTERS))
  def test_residual_block_call(self, batch_size, image_height, image_width,
                               image_channel, num_filters):
    inputs = _generate_data_batch(batch_size, image_height, image_width,
                                  image_channel)

    residual_block = resunet_extractor_lib.ResidualBlock(
        num_filters=num_filters)

    output = residual_block(inputs)

    self.assertSequenceEqual(
        output.shape, (batch_size, image_height, image_width, num_filters))


if __name__ == '__main__':
  tf.test.main()
