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

"""Tests for google3.vr.perception.deepholodeck.human_correspondence.train_eval_lib."""

import functools
import tempfile
from unittest import mock

from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow as tf

from google3.vr.perception.deepholodeck.human_correspondence import dataset_lib
from google3.vr.perception.deepholodeck.human_correspondence import train_eval_lib
from google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor.model import geodesic_feature_network
from google3.vr.perception.deepholodeck.human_correspondence.optical_flow.pwcnet import pwcnet_lib
from google3.vr.perception.deepholodeck.human_correspondence.optical_flow.raft import raft_lib


def _generate_small_test_dataset():
  """Generates a data like from the data_lib.load_dataset()."""
  features = {}
  features['images'] = tf.ones((1, 2, 32, 32, 3), dtype=tf.float32)
  features['flows'] = tf.ones((1, 32, 32, 2), dtype=tf.float32)
  features['masks'] = tf.ones((1, 2, 32, 32, 1), dtype=tf.bool)
  features['flow_mask'] = tf.ones((1, 32, 32, 1), dtype=tf.bool)
  features['source_geo_centers'] = tf.ones((1, 2, 2), dtype=tf.float32)
  features['target_geo_centers'] = tf.ones((1, 2, 2), dtype=tf.float32)
  features['source_geodesic_maps'] = tf.ones((1, 32, 32, 2), dtype=tf.float32)
  features['target_geodesic_maps'] = tf.ones((1, 32, 32, 2), dtype=tf.float32)
  features['source_cross_geo_centers'] = tf.ones((1, 8, 2), dtype=tf.float32)
  features['target_cross_geo_centers'] = tf.ones((1, 8, 2), dtype=tf.float32)
  features['source_cross_geodesic_maps'] = tf.ones((1, 32, 32, 1),
                                                   dtype=tf.float32)
  features['target_cross_geodesic_maps'] = tf.ones((1, 32, 32, 1),
                                                   dtype=tf.float32)
  features['source_geo_keypoints'] = tf.ones((1, 6, 2), dtype=tf.float32)
  features['target_geo_keypoints'] = tf.ones((1, 6, 2), dtype=tf.float32)
  features['source_geodesic_matrix'] = tf.ones((1, 3, 3, 1), dtype=tf.float32)
  features['target_geodesic_matrix'] = tf.ones((1, 3, 3, 1), dtype=tf.float32)
  features['has_no_geodesic'] = tf.zeros((1,), dtype=tf.bool)
  yield features


def _generate_large_test_dataset():
  """Generates a data like from the data_lib.load_dataset()."""
  features = {}
  features['images'] = tf.ones((1, 2, 64, 64, 3), dtype=tf.float32)
  features['flows'] = tf.ones((1, 64, 64, 2), dtype=tf.float32)
  features['masks'] = tf.ones((1, 2, 64, 64, 1), dtype=tf.bool)
  features['flow_mask'] = tf.ones((1, 64, 64, 1), dtype=tf.bool)
  features['source_geo_centers'] = tf.ones((1, 2, 2), dtype=tf.float32)
  features['target_geo_centers'] = tf.ones((1, 2, 2), dtype=tf.float32)
  features['source_geodesic_maps'] = tf.ones((1, 64, 64, 2), dtype=tf.float32)
  features['target_geodesic_maps'] = tf.ones((1, 64, 64, 2), dtype=tf.float32)
  features['source_cross_geo_centers'] = tf.ones((1, 8, 2), dtype=tf.float32)
  features['target_cross_geo_centers'] = tf.ones((1, 8, 2), dtype=tf.float32)
  features['source_cross_geodesic_maps'] = tf.ones((1, 64, 64, 1),
                                                   dtype=tf.float32)
  features['target_cross_geodesic_maps'] = tf.ones((1, 64, 64, 1),
                                                   dtype=tf.float32)
  features['source_geo_keypoints'] = tf.ones((1, 6, 2), dtype=tf.float32)
  features['target_geo_keypoints'] = tf.ones((1, 6, 2), dtype=tf.float32)
  features['source_geodesic_matrix'] = tf.ones((1, 3, 3, 1), dtype=tf.float32)
  features['target_geodesic_matrix'] = tf.ones((1, 3, 3, 1), dtype=tf.float32)
  features['has_no_geodesic'] = tf.zeros((1,), dtype=tf.bool)
  yield features


def _get_test_dataset_types():
  """Generates the type for data from data_lib.load_dataset()."""
  types = {
      'images': tf.float32,
      'flows': tf.float32,
      'masks': tf.bool,
      'flow_mask': tf.bool,
      'source_geo_centers': tf.float32,
      'target_geo_centers': tf.float32,
      'source_geodesic_maps': tf.float32,
      'target_geodesic_maps': tf.float32,
      'source_cross_geo_centers': tf.float32,
      'target_cross_geo_centers': tf.float32,
      'source_cross_geodesic_maps': tf.float32,
      'target_cross_geodesic_maps': tf.float32,
      'source_geo_keypoints': tf.float32,
      'target_geo_keypoints': tf.float32,
      'source_geodesic_matrix': tf.float32,
      'target_geodesic_matrix': tf.float32,
      'has_no_geodesic': tf.bool,
  }
  return types


class TrainLibTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(
      (train_eval_lib.TrainingMode.CPU, tf.distribute.OneDeviceStrategy),
      (train_eval_lib.TrainingMode.GPU, tf.distribute.MirroredStrategy))
  def test_get_strategy(self, training_mode, return_type):
    # tpu mode cannot be tested without specific environment.
    strategy = train_eval_lib.get_strategy(training_mode, 'dummy')

    self.assertIsInstance(strategy, return_type)

  def test_OneCycleLR(self):
    lr_schedule = train_eval_lib.OneCycleLR(0.1, 0.1, 200, 0)

    self.assertIsInstance(lr_schedule,
                          tf.keras.optimizers.schedules.LearningRateSchedule)

  @parameterized.parameters(
      dict(
          lr_params={'piecewise_lr': {
              'learning_rate': 1e-4,
              'start_steps': 0
          }}),
      dict(
          lr_params={
              'one_cycle_lr': {
                  'learning_rate': 4e-4,
                  'start_steps': 0,
                  'max_train_steps': 300000
              }
          }))
  @flagsaver.flagsaver(xm_runlocal=True)
  @mock.patch.object(dataset_lib, 'load_dataset', autospec=True)
  @mock.patch.object(train_eval_lib, 'get_training_elements', autospec=True)
  def test_onedevice_strategy_for_train(self, mock_get_training_elements,
                                        mock_dataset_loader, lr_params):
    mock_dataset_loader.return_value = tf.data.Dataset.from_generator(
        _generate_small_test_dataset, output_types=_get_test_dataset_types())

    base_folder = tempfile.mkdtemp()
    model_hparams = {
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
        'filters_sequence': (2, 2, 2, 2, 2),
        'train_search_size': (3, 3),
        'eval_search_size': (3, 3),
    }

    mock_get_training_elements.return_value = (
        functools.partial(geodesic_feature_network.GeoFeatureNet,
                          model_hparams))

    # Smoke test to make sure that the model builds and trains for an epoch.
    tf.config.run_functions_eagerly(True)
    train_eval_lib.train_pipeline(
        training_mode='cpu',
        tpu_bns='',
        base_folder=base_folder,
        dataset_params={},
        lr_params=lr_params,
        batch_size=1,
        n_iterations=1,
        save_every_n_batches=1,
        time_every_n_steps=1)

    # Smoke test to make sure that the model can be evaluated.
    train_eval_lib.eval_pipeline(
        eval_mode='cpu',
        tpu_bns='',
        dataset_params={'test': {}},
        train_base_folder=base_folder,
        eval_base_folder=base_folder,
        batch_size=1,
        eval_name='test')

  @flagsaver.flagsaver(xm_runlocal=True)
  @mock.patch.object(dataset_lib, 'load_dataset', autospec=True)
  @mock.patch.object(train_eval_lib, 'get_training_elements', autospec=True)
  def test_onedevice_strategy_for_train_pwcnet(self, mock_get_training_elements,
                                               mock_dataset_loader):
    lr_params = {
        'one_cycle_lr': {
            'learning_rate': 4e-4,
            'start_steps': 0,
            'max_train_steps': 300000
        }
    }
    mock_dataset_loader.return_value = tf.data.Dataset.from_generator(
        _generate_small_test_dataset, output_types=_get_test_dataset_types())

    base_folder = tempfile.mkdtemp()
    model_hparams = {
        'loss_hparams': {
            'correspondence_loss_weight': 1.0,
            'triplet_loss_weight': 0.2,
            'dense_geodesic_loss_weight': 3.0,
            'cross_dense_geodesic_loss_weight': 3.0,
            'sparse_ordinal_geodesic_loss_weight': 5.0,
            'optical_flow_loss_weight': 1.0,
        },
        'encoder_filters_sequence': (None, 16, 16, 16, 16, 16),
        'decoder_filters_sequence': (16, 16, 16, 16, 16),
        'prediction_level': 0,
        'cost_volume_range': 4,
        'use_dense_connections': True,
        'flow_scale_factor': 20,
        'scale_upsampled_flow': True,
        'path_drop_probabilities': [1.0, 1.0],
        'use_normalized_feature': True,
        'has_pred_mask': True,
        'train_search_size': (3, 3),
        'eval_search_size': (3, 3),
    }

    mock_get_training_elements.return_value = (
        functools.partial(pwcnet_lib.PWCNet, model_hparams))

    # Smoke test to make sure that the model builds and trains for an epoch.
    tf.config.run_functions_eagerly(True)
    train_eval_lib.train_pipeline(
        training_mode='cpu',
        tpu_bns='',
        base_folder=base_folder,
        dataset_params={},
        lr_params=lr_params,
        batch_size=1,
        n_iterations=1,
        save_every_n_batches=1,
        time_every_n_steps=1)

  @flagsaver.flagsaver(xm_runlocal=True)
  @mock.patch.object(dataset_lib, 'load_dataset', autospec=True)
  @mock.patch.object(train_eval_lib, 'get_training_elements', autospec=True)
  def test_onedevice_strategy_for_train_raft(self, mock_get_training_elements,
                                             mock_dataset_loader):
    lr_params = {
        'one_cycle_lr': {
            'learning_rate': 4e-4,
            'start_steps': 0,
            'max_train_steps': 300000
        }
    }
    mock_dataset_loader.return_value = tf.data.Dataset.from_generator(
        _generate_large_test_dataset, output_types=_get_test_dataset_types())

    base_folder = tempfile.mkdtemp()
    model_hparams = {
        'loss_hparams': {
            'correspondence_loss_weight': 1.0,
            'triplet_loss_weight': 0.2,
            'dense_geodesic_loss_weight': 3.0,
            'cross_dense_geodesic_loss_weight': 3.0,
            'sparse_ordinal_geodesic_loss_weight': 5.0,
            'optical_flow_loss_weight': 1.0,
        },
        'flow_scale_factor': 20,
        'max_rec_iters': 12,
        'drop_out': 0.0,
        'use_norms': True,
        'small': False,
        'path_drop_probabilities': [1.0, 1.0],
        'train_search_size': (3, 3),
        'eval_search_size': (3, 3),
    }

    mock_get_training_elements.return_value = (
        functools.partial(raft_lib.RAFT, model_hparams))

    # Smoke test to make sure that the model builds and trains for an epoch.
    tf.config.run_functions_eagerly(True)
    train_eval_lib.train_pipeline(
        training_mode='cpu',
        tpu_bns='',
        base_folder=base_folder,
        dataset_params={},
        lr_params=lr_params,
        batch_size=1,
        n_iterations=1,
        save_every_n_batches=1,
        time_every_n_steps=1)


if __name__ == '__main__':
  tf.test.main()
