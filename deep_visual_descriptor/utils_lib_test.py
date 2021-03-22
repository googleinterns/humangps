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

"""Tests for google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor.utils_lib."""
from absl.testing import parameterized
import tensorflow as tf

from google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor import utils_lib

_FEATURE_MAP_SIZE = (12, 14)

_FEATURE_DIMS = 2
_BATCH = 2
_SOURCE_SAMPLE_SIZE = 10
_TARGET_SAMPLE_SIZE = 5
_NUM_SAMPLE = 15


class UtilsLibTest(parameterized.TestCase, tf.test.TestCase):

  def test_compute_l2_distance_matrix(self):
    source_sample = tf.random.uniform((
        _SOURCE_SAMPLE_SIZE,
        _FEATURE_DIMS,
    ),
                                      dtype=tf.float32)
    target_sample = tf.random.uniform((
        _TARGET_SAMPLE_SIZE,
        _FEATURE_DIMS,
    ),
                                      dtype=tf.float32)

    distance = utils_lib.compute_l2_distance_matrix(source_sample,
                                                    target_sample)

    self.assertSequenceEqual(distance.shape,
                             (_SOURCE_SAMPLE_SIZE, _TARGET_SAMPLE_SIZE))

  def test_compute_cosine_distance_matrix(self):
    source_sample = tf.random.uniform((
        _SOURCE_SAMPLE_SIZE,
        _FEATURE_DIMS,
    ),
                                      dtype=tf.float32)
    target_sample = tf.random.uniform((
        _TARGET_SAMPLE_SIZE,
        _FEATURE_DIMS,
    ),
                                      dtype=tf.float32)

    distance = utils_lib.compute_cosine_distance_matrix(source_sample,
                                                        target_sample)

    self.assertSequenceEqual(distance.shape,
                             (_SOURCE_SAMPLE_SIZE, _TARGET_SAMPLE_SIZE))

  @parameterized.parameters('L2', 'cosine')
  def test_find_correspondence_by_nearest_neighbor(self, method):
    src_feature_map = tf.random.uniform(
        _FEATURE_MAP_SIZE + (_FEATURE_DIMS,), dtype=tf.float32)
    tgt_feature_map = tf.random.uniform(
        _FEATURE_MAP_SIZE + (_FEATURE_DIMS,), dtype=tf.float32)
    src_mask = tf.ones(_FEATURE_MAP_SIZE + (1,), dtype=tf.bool)
    tgt_mask = tf.ones(_FEATURE_MAP_SIZE + (1,), dtype=tf.bool)

    correspondence, distance_map = utils_lib.find_correspondence_by_nearest_neighbor(
        src_feature_map, tgt_feature_map, src_mask, tgt_mask, method)

    self.assertSequenceEqual(correspondence.shape, _FEATURE_MAP_SIZE + (2,))
    self.assertSequenceEqual(distance_map.shape, _FEATURE_MAP_SIZE + (1,))

  @parameterized.parameters('L2', 'cosine')
  def test_batch_find_correspondence_by_nearest_search(self, method):
    src_feature_maps = tf.random.uniform(
        (_BATCH,) + _FEATURE_MAP_SIZE + (_FEATURE_DIMS,), dtype=tf.float32)
    tgt_feature_maps = tf.random.uniform(
        (_BATCH,) + _FEATURE_MAP_SIZE + (_FEATURE_DIMS,), dtype=tf.float32)
    src_masks = tf.ones((_BATCH,) + _FEATURE_MAP_SIZE + (1,), dtype=tf.bool)
    tgt_masks = tf.ones((_BATCH,) + _FEATURE_MAP_SIZE + (1,), dtype=tf.bool)

    correspondence, distance_map = utils_lib.batch_find_correspondence_by_nearest_search(
        src_feature_maps, tgt_feature_maps, src_masks, tgt_masks,
        _FEATURE_MAP_SIZE, method)

    self.assertSequenceEqual(correspondence.shape,
                             (_BATCH,) + _FEATURE_MAP_SIZE + (2,))
    self.assertSequenceEqual(distance_map.shape,
                             (_BATCH,) + _FEATURE_MAP_SIZE + (1,))

  def test_sparse_bilinear_sample(self):
    image = tf.random.uniform(
        (_BATCH,) + _FEATURE_MAP_SIZE + (_FEATURE_DIMS,), dtype=tf.float32)
    xy_coords = tf.random.uniform((_BATCH, _NUM_SAMPLE, 2), dtype=tf.float32)
    height_coords = _FEATURE_MAP_SIZE[0]
    width_coords = _FEATURE_MAP_SIZE[1]
    pad_mode = 'EDGE'

    output = utils_lib.sparse_bilinear_sample(image, xy_coords, height_coords,
                                              width_coords, pad_mode)

    self.assertSequenceEqual(output.shape, (_BATCH, _NUM_SAMPLE, _FEATURE_DIMS))

  def test_dense_bilinear_sample(self):
    image = tf.random.uniform(
        (_BATCH,) + _FEATURE_MAP_SIZE + (_FEATURE_DIMS,), dtype=tf.float32)
    flow = tf.random.uniform(
        (_BATCH,) + _FEATURE_MAP_SIZE + (_FEATURE_DIMS,), dtype=tf.float32)
    pad_mode = 'EDGE'

    output = utils_lib.dense_bilinear_sample(image, flow, pad_mode)

    self.assertSequenceEqual(output.shape,
                             (_BATCH,) + _FEATURE_MAP_SIZE + (_FEATURE_DIMS,))


if __name__ == '__main__':
  tf.test.main()
