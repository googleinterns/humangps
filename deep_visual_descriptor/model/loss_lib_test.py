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

"""Tests for google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor.model.loss_lib."""

import tensorflow as tf

from google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor.model import loss_lib


def _get_test_data_batch():
  input_batch = {}
  input_batch['images'] = tf.ones((1, 2, 16, 16, 3), dtype=tf.float32)
  input_batch['flows'] = tf.ones((1, 16, 16, 2), dtype=tf.float32)
  input_batch['masks'] = tf.ones((1, 2, 16, 16, 1), dtype=tf.bool)
  input_batch['flow_mask'] = tf.ones((1, 16, 16, 1), dtype=tf.bool)
  input_batch['source_geo_centers'] = tf.ones((1, 2, 2), dtype=tf.float32)
  input_batch['target_geo_centers'] = tf.ones((1, 2, 2), dtype=tf.float32)
  input_batch['source_geodesic_maps'] = tf.ones((1, 16, 16, 2),
                                                dtype=tf.float32)
  input_batch['target_geodesic_maps'] = tf.ones((1, 16, 16, 2),
                                                dtype=tf.float32)
  input_batch['source_cross_geo_centers'] = tf.ones((1, 8, 2), dtype=tf.float32)
  input_batch['target_cross_geo_centers'] = tf.ones((1, 8, 2), dtype=tf.float32)
  input_batch['source_cross_geodesic_maps'] = tf.ones((1, 16, 16, 1),
                                                      dtype=tf.float32)
  input_batch['target_cross_geodesic_maps'] = tf.ones((1, 16, 16, 1),
                                                      dtype=tf.float32)
  input_batch['source_geo_keypoints'] = tf.ones((1, 6, 2), dtype=tf.float32)
  input_batch['target_geo_keypoints'] = tf.ones((1, 6, 2), dtype=tf.float32)
  input_batch['source_geodesic_matrix'] = tf.ones((1, 3, 3, 1),
                                                  dtype=tf.float32)
  input_batch['target_geodesic_matrix'] = tf.ones((1, 3, 3, 1),
                                                  dtype=tf.float32)
  input_batch['has_no_geodesic'] = tf.zeros((1,), dtype=tf.bool)
  return input_batch


def _get_loss_hparams():
  loss_hparams = {
      'correspondence_loss_weight': 1.0,
      'triplet_loss_weight': 0.0,
      'dense_geodesic_loss_weight': 0.0,
      'cross_dense_geodesic_loss_weight': 0.0,
      'sparse_ordinal_geodesic_loss_weight': 0.0,
  }
  return loss_hparams


class LossLibTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.input_batch = _get_test_data_batch()
    self.loss_hparams = _get_loss_hparams()

  def test_geodesic_feature_loss(self):
    tolerance = 1e-5
    source_feature_pyramid = [
        tf.ones((1, 16, 16, 1)),
        tf.ones((1, 16, 16, 1)),
        tf.ones((1, 8, 8, 1)),
        tf.ones((1, 8, 8, 1)),
        tf.ones((1, 4, 4, 1)),
        tf.ones((1, 4, 4, 1)),
    ]
    target_feature_pyramid = [
        tf.ones((1, 16, 16, 1)),
        tf.ones((1, 16, 16, 1)),
        tf.ones((1, 8, 8, 1)),
        tf.ones((1, 8, 8, 1)),
        tf.ones((1, 4, 4, 1)),
        tf.ones((1, 4, 4, 1)),
    ]

    loss_model = loss_lib.GeodesicFeatureLoss(self.loss_hparams)
    loss_model.set_coord_size(16, 16)

    loss_summaries = loss_model.get_training_loss(source_feature_pyramid,
                                                  target_feature_pyramid,
                                                  self.input_batch)

    for key, _ in self.loss_hparams.items():
      self.assertIn(key[:-7], list(loss_summaries.keys()))
    self.assertIn('training_loss', list(loss_summaries.keys()))
    self.assertAlmostEqual(loss_summaries['training_loss'], 0, delta=tolerance)
    self.assertAlmostEqual(loss_model._height_coords, 16, delta=tolerance)
    self.assertAlmostEqual(loss_model._width_coords, 16, delta=tolerance)

  def test_get_average_epe(self):
    tolerance = 1e-5
    flow_pred = tf.ones((1, 16, 16, 2), dtype=tf.float32)
    flow_gt = tf.ones((1, 16, 16, 2), dtype=tf.float32)
    flow_mask = tf.ones((1, 2, 16, 16, 1), dtype=tf.bool)

    loss = loss_lib.GeodesicFeatureLoss.get_average_epe(flow_pred, flow_gt,
                                                        flow_mask)

    self.assertAlmostEqual(loss, 0.0, delta=tolerance)


if __name__ == '__main__':
  tf.test.main()
