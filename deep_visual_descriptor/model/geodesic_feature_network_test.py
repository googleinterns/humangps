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

"""Tests for google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor.model.geodesic_feature_network."""
import tensorflow as tf

from google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor.model import geodesic_feature_network


def _get_test_data_batch():
  input_batch = {}
  input_batch['images'] = tf.ones((1, 2, 64, 64, 3), dtype=tf.float32)
  input_batch['flows'] = tf.ones((1, 64, 64, 2), dtype=tf.float32)
  input_batch['masks'] = tf.ones((1, 2, 64, 64, 1), dtype=tf.bool)
  input_batch['flow_mask'] = tf.ones((1, 64, 64, 1), dtype=tf.bool)
  input_batch['source_geo_centers'] = tf.ones((1, 2, 2), dtype=tf.float32)
  input_batch['target_geo_centers'] = tf.ones((1, 2, 2), dtype=tf.float32)
  input_batch['source_geodesic_maps'] = tf.ones((1, 64, 64, 2),
                                                dtype=tf.float32)
  input_batch['target_geodesic_maps'] = tf.ones((1, 64, 64, 2),
                                                dtype=tf.float32)
  input_batch['source_cross_geo_centers'] = tf.ones((1, 8, 2), dtype=tf.float32)
  input_batch['target_cross_geo_centers'] = tf.ones((1, 8, 2), dtype=tf.float32)
  input_batch['source_cross_geodesic_maps'] = tf.ones((1, 64, 64, 1),
                                                      dtype=tf.float32)
  input_batch['target_cross_geodesic_maps'] = tf.ones((1, 64, 64, 1),
                                                      dtype=tf.float32)
  input_batch['source_geo_keypoints'] = tf.ones((1, 6, 2), dtype=tf.float32)
  input_batch['target_geo_keypoints'] = tf.ones((1, 6, 2), dtype=tf.float32)
  input_batch['source_geodesic_matrix'] = tf.ones((1, 3, 3, 1),
                                                  dtype=tf.float32)
  input_batch['target_geodesic_matrix'] = tf.ones((1, 3, 3, 1),
                                                  dtype=tf.float32)
  input_batch['has_no_geodesic'] = tf.zeros((1,), dtype=tf.bool)
  return input_batch


def _get_model_hparams():
  return {
      'loss_hparams': {
          'correspondence_loss_weight': 0.0,
          'triplet_loss_weight': 0.0,
          'dense_geodesic_loss_weight': 0.0,
          'cross_dense_geodesic_loss_weight': 0.0,
          'sparse_ordinal_geodesic_loss_weight': 0.0,
      },
      'flow_scale_factor': 20.0,
      'return_feature_pyramid': True,
      'feature_extractor': 'resunet',
      'filters_sequence': (1, 1, 1, 1, 1, 1, 1)
  }


class GeodesicFeatureNetworkTest(tf.test.TestCase):

  def setUp(self):
    super().setUp()
    self.input_batch = _get_test_data_batch()
    self.model_hparams = _get_model_hparams()

  def test_get_train_outputs(self):
    tolerance = 1e-9
    model = geodesic_feature_network.GeoFeatureNet(self.model_hparams)

    training_loss, scalar_summaries, image_summaries = model.get_train_outputs(
        self.input_batch)

    with self.subTest(name='check_training_loss'):
      self.assertAlmostEqual(training_loss, 0.0, delta=tolerance)
      self.assertIn('training_loss', list(scalar_summaries.keys()))
      self.assertIn('sparse_ordinal_geodesic_loss',
                    list(scalar_summaries.keys()))
      self.assertIn('cross_dense_geodesic_loss', list(scalar_summaries.keys()))
      self.assertIn('dense_geodesic_loss', list(scalar_summaries.keys()))
      self.assertIn('triplet_loss', list(scalar_summaries.keys()))
      self.assertIn('correspondence_loss', list(scalar_summaries.keys()))

    with self.subTest(name='check_metric_summaries'):
      self.assertIn('all_aepe_search', list(scalar_summaries.keys()))
      self.assertIn('masked_aepe_search', list(scalar_summaries.keys()))

    with self.subTest(name='check_first_image_summary'):
      self.assertIn('first_image', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['first_image'].shape,
                               (1, 64, 64, 3))
    with self.subTest(name='check_second_image_summary'):
      self.assertIn('second_image', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['second_image'].shape,
                               (1, 64, 64, 3))
    with self.subTest(name='check_gt_flow_summary'):
      self.assertIn('gt_flow', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['gt_flow'].shape, (1, 64, 64, 3))
    with self.subTest(name='check_gt_warped_image_summary'):
      self.assertIn('gt_warped_image', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['gt_warped_image'].shape,
                               (1, 64, 64, 3))
    with self.subTest(name='check_gt_diff_image_summary'):
      self.assertIn('gt_diff_image', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['gt_diff_image'].shape,
                               (1, 64, 64, 3))
    with self.subTest(name='check_warped_image_search_summary'):
      self.assertIn('warped_image_search', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['warped_image_search'].shape,
                               (1, 64, 64, 3))
    with self.subTest(name='check_flow_search_summary'):
      self.assertIn('flow_search', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['flow_search'].shape,
                               (1, 64, 64, 3))
    with self.subTest(name='check_diff_image_search_summary'):
      self.assertIn('diff_image_search', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['diff_image_search'].shape,
                               (1, 64, 64, 3))

  def test_get_eval_outputs(self):
    tolerance = 1e-9
    model = geodesic_feature_network.GeoFeatureNet(self.model_hparams)

    total_loss, scalar_summaries, image_summaries = model.get_eval_outputs(
        self.input_batch)

    with self.subTest(name='check_training_loss'):
      self.assertAlmostEqual(total_loss, 0.0, delta=tolerance)

    with self.subTest(name='check_metric_summaries'):
      self.assertIn('all_aepe_search', list(scalar_summaries.keys()))
      self.assertIn('masked_aepe_search', list(scalar_summaries.keys()))

    with self.subTest(name='check_first_image_summary'):
      self.assertIn('first_image', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['first_image'].shape,
                               (1, 64, 64, 3))
    with self.subTest(name='check_second_image_summary'):
      self.assertIn('second_image', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['second_image'].shape,
                               (1, 64, 64, 3))
    with self.subTest(name='check_gt_flow_summary'):
      self.assertIn('gt_flow', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['gt_flow'].shape, (1, 64, 64, 3))
    with self.subTest(name='check_gt_warped_image_summary'):
      self.assertIn('gt_warped_image', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['gt_warped_image'].shape,
                               (1, 64, 64, 3))
    with self.subTest(name='check_gt_diff_image_summary'):
      self.assertIn('gt_diff_image', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['gt_diff_image'].shape,
                               (1, 64, 64, 3))
    with self.subTest(name='check_warped_image_search_summary'):
      self.assertIn('warped_image_search', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['warped_image_search'].shape,
                               (1, 64, 64, 3))
    with self.subTest(name='check_flow_search_summary'):
      self.assertIn('flow_search', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['flow_search'].shape,
                               (1, 64, 64, 3))
    with self.subTest(name='check_diff_image_search_summary'):
      self.assertIn('diff_image_search', list(image_summaries.keys()))
      self.assertSequenceEqual(image_summaries['diff_image_search'].shape,
                               (1, 64, 64, 3))


if __name__ == '__main__':
  tf.test.main()
