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

"""Tests for google3.vr.perception.deepholodeck.human_correspondence.dataset_lib."""
import tensorflow as tf

from google3.vr.perception.deepholodeck.human_correspondence import dataset_lib

_IMAGE_HEIGHT = 12
_IMAGE_WIDTH = 16
_NUM_MAP_SAMPLE = 4
_NUM_SPARSE_SAMPLE = 8


def _get_sstable_data():
  """Get an expected data like from an sstable."""
  example = {}
  for key in ['source_rgb', 'target_rgb']:
    example[key] = tf.ones((_IMAGE_HEIGHT, _IMAGE_WIDTH, 3), dtype=tf.uint16)
  example['forward_flow'] = tf.ones((_IMAGE_HEIGHT, _IMAGE_WIDTH, 2),
                                    dtype=tf.uint16)
  for key in ['source_mask', 'target_mask', 'flow_mask']:
    example[key] = tf.ones((_IMAGE_HEIGHT, _IMAGE_WIDTH, 1), dtype=tf.uint8)
  for key in [
      'source_geo_centers', 'target_geo_centers', 'source_cross_geo_centers',
      'target_cross_geo_centers'
  ]:
    example[key] = tf.ones((_NUM_MAP_SAMPLE, 2), dtype=tf.float32)
  for key in [
      'source_geodesic_maps', 'target_geodesic_maps',
      'source_cross_geodesic_maps', 'target_cross_geodesic_maps'
  ]:
    example[key] = tf.ones((_IMAGE_HEIGHT, _IMAGE_WIDTH, _NUM_MAP_SAMPLE),
                           dtype=tf.uint16)
  for key in ['source_geo_keypoints', 'target_geo_keypoints']:
    example[key] = tf.ones((_NUM_SPARSE_SAMPLE, 2), dtype=tf.float32)
  for key in ['source_geodesic_matrix', 'target_geodesic_matrix']:
    example[key] = tf.ones(
        (_NUM_SPARSE_SAMPLE // 2, _NUM_SPARSE_SAMPLE // 2, 1), dtype=tf.uint16)
  return example


def _get_processed_data():
  example = {}
  example['images'] = tf.ones((2, _IMAGE_HEIGHT, _IMAGE_WIDTH, 3),
                              dtype=tf.float32)
  example['flows'] = tf.ones((_IMAGE_HEIGHT, _IMAGE_WIDTH, 2), dtype=tf.float32)
  example['masks'] = tf.ones((2, _IMAGE_HEIGHT, _IMAGE_WIDTH, 1), dtype=tf.bool)
  example['flow_mask'] = tf.ones((_IMAGE_HEIGHT, _IMAGE_WIDTH, 1),
                                 dtype=tf.bool)
  example['source_geo_centers'] = tf.ones((_NUM_MAP_SAMPLE, 2),
                                          dtype=tf.float32)
  example['target_geo_centers'] = tf.ones((_NUM_MAP_SAMPLE, 2),
                                          dtype=tf.float32)
  example['source_geodesic_maps'] = tf.ones(
      (_IMAGE_HEIGHT, _IMAGE_WIDTH, _NUM_MAP_SAMPLE), dtype=tf.float32)
  example['target_geodesic_maps'] = tf.ones(
      (_IMAGE_HEIGHT, _IMAGE_WIDTH, _NUM_MAP_SAMPLE), dtype=tf.float32)
  example['source_cross_geo_centers'] = tf.ones((_NUM_MAP_SAMPLE, 2),
                                                dtype=tf.float32)
  example['target_cross_geo_centers'] = tf.ones((_NUM_MAP_SAMPLE, 2),
                                                dtype=tf.float32)
  example['source_cross_geodesic_maps'] = tf.ones(
      (_IMAGE_HEIGHT, _IMAGE_WIDTH, _NUM_MAP_SAMPLE), dtype=tf.float32)
  example['target_cross_geodesic_maps'] = tf.ones(
      (_IMAGE_HEIGHT, _IMAGE_WIDTH, _NUM_MAP_SAMPLE), dtype=tf.float32)
  example['source_geo_keypoints'] = tf.ones((_NUM_SPARSE_SAMPLE, 2),
                                            dtype=tf.float32)
  example['target_geo_keypoints'] = tf.ones((_NUM_SPARSE_SAMPLE, 2),
                                            dtype=tf.float32)
  example['source_geodesic_matrix'] = tf.ones(
      (_NUM_SPARSE_SAMPLE // 2, _NUM_SPARSE_SAMPLE // 2, 1), dtype=tf.float32)
  example['target_geodesic_matrix'] = tf.ones(
      (_NUM_SPARSE_SAMPLE // 2, _NUM_SPARSE_SAMPLE // 2, 1), dtype=tf.float32)
  example['has_no_geodesic'] = tf.zeros(1, dtype=tf.bool)
  return example


class DatasetLibTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self._processed_example = _get_processed_data()
    self._example = _get_sstable_data()

  def test_augment_colors(self):
    augmented_example = dataset_lib.augment_colors(self._processed_example)
    images = self._processed_example['images']

    self.assertSequenceEqual(augmented_example['images'].shape, images.shape)

  def test__preprocess(self):
    attributes = ()
    processed_example = dataset_lib._preprocess(self._example, attributes)

    for key, value in processed_example.items():
      if key != 'has_no_geodesic':
        self.assertSequenceEqual(value.shape,
                                 self._processed_example[key].shape)


if __name__ == '__main__':
  tf.test.main()
