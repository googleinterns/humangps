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

"""Library defining the architecture of geodesic feature learning model."""

from typing import Any, Dict, Tuple, Sequence

import tensorflow as tf

from google3.research.vision.piedpiper.brain.python.ops import flow_ops
from google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor import utils_lib as descriptor_utils
from google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor.model import loss_lib
from google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor.network import resunet_extractor_lib

_UINT_16 = 65535
_EPSILON = 1e-12
_SEARCH_SIZE = (192, 128)


class GeoFeatureNet(tf.keras.Model):
  """Build a geodesic feature learning model for dense human correspondence."""

  def __init__(self,
               model_hparams: Dict[str, Any] = None,
               name: str = 'GeoFeatureNet',
               **kwargs):
    """Initialize the geodesic feature learning model.

    Args:
      model_hparams: The paarameter for the model.
      name: Model name.
      **kwargs: Keyworded arguments that are forwarded by the model.
    """
    super(GeoFeatureNet, self).__init__(name=name, **kwargs)
    self._loss_hparams = model_hparams['loss_hparams']
    self._flow_scale_factor = model_hparams['flow_scale_factor']
    self._return_feature_pyramid = model_hparams['return_feature_pyramid']

    if 'train_search_size' in model_hparams:
      self._train_search_size = model_hparams['train_search_size']
    else:
      self._train_search_size = _SEARCH_SIZE
    if 'eval_search_size' in model_hparams:
      self._eval_search_size = model_hparams['eval_search_size']
    else:
      self._eval_search_size = _SEARCH_SIZE

    self._loss_module = loss_lib.GeodesicFeatureLoss(self._loss_hparams)

    if model_hparams['feature_extractor'] == 'resunet':
      self.feature_extractor = resunet_extractor_lib.ResUNet(
          model_hparams['filters_sequence'],
          return_feature_pyramid=self._return_feature_pyramid)

  @classmethod
  def normalize_feature_pyramid(
      cls, features: Sequence[tf.Tensor]) -> Sequence[tf.Tensor]:
    """Normalize all the feature of a feature pyramid in channel dimension."""
    normalized_features = []
    for feature in features:
      normalized_feature = tf.math.l2_normalize(
          feature, axis=-1, epsilon=_EPSILON)
      normalized_features.append(normalized_feature)
    return normalized_features

  def call(self,
           image_pair: tf.Tensor,
           training: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
    """Run the forward pass of geodesic feature learning model.

    Args:
      image_pair: The input image pair to the network of size B x 2 x H x W x 3,
        where B is the number of batch size, H x W is the size of input images.
      training: A boolean that defines whether the call should be excuted as a
        training or an inference call.

    Returns:
      A tuple of source and target feature map for source and target input
      image.
    """
    source_image, target_image = tf.unstack(image_pair, axis=1)[:2]
    images = tf.concat([source_image, target_image], axis=0)
    if self._return_feature_pyramid:
      feature_maps, feature_pyramids = self.feature_extractor(images, training)
    else:
      feature_maps = self.feature_extractor(images, training)

    feature_maps = tf.math.l2_normalize(feature_maps, axis=-1, epsilon=_EPSILON)
    source_feature_map, target_feature_map = tf.split(
        feature_maps, num_or_size_splits=2, axis=0)
    self.feature_map_pair = (source_feature_map, target_feature_map)

    if self._return_feature_pyramid:
      feature_pyramids = self.normalize_feature_pyramid(feature_pyramids)
      feature_pyramids = [
          tf.split(feature_map, num_or_size_splits=2, axis=0)
          for feature_map in feature_pyramids
      ]
      source_feature_pyramid = [
          feature_map[0] for feature_map in feature_pyramids
      ]
      target_feature_pyramid = [
          feature_map[1] for feature_map in feature_pyramids
      ]
      self.feature_pyramid_pair = (source_feature_pyramid,
                                   target_feature_pyramid)
    else:
      self.feature_pyramid_pair = ([source_feature_map], [target_feature_map])

    return self.feature_map_pair

  def get_train_outputs(
      self, input_batch: Dict[str, tf.Tensor]
  ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Run the geodesic feature learning model for training.

    Args:
      input_batch: A dictionary holding the data batch.

    Returns:
      training_loss: A scalar tensor.
      scalar_summaries: A dictionary holding scalar tensors to summarize.
      image_summaries: A dictionary holding image tensors to summarize.
    """
    image_pair = input_batch['images']
    feature_map_pair = self.call(image_pair)

    if 'flow_mask' not in input_batch:
      input_batch['flow_mask'] = tf.math.less(
          tf.reduce_mean(tf.abs(input_batch['flows']), axis=-1), _UINT_16)
    if 'masks' not in input_batch:
      masks = input_batch['flow_mask'][:, tf.newaxis, :, :, :]
      input_batch['masks'] = tf.tile(masks, [1, 2, 1, 1, 1])
    input_batch['flows'] = input_batch['flows'] * self._flow_scale_factor

    _, _, image_height, image_width, _ = image_pair.get_shape().as_list()
    self._loss_module.set_coord_size(image_height, image_width)
    loss_summaries = self._loss_module.get_training_loss(
        self.feature_pyramid_pair[0], self.feature_pyramid_pair[1], input_batch)

    flow_gt = input_batch['flows']
    flow_mask = tf.cast(input_batch['flow_mask'], tf.float32)
    source_mask = input_batch['masks'][:, 0]
    target_mask = input_batch['masks'][:, 1]
    (correspondence_maps,
     _) = descriptor_utils.batch_find_correspondence_by_nearest_search(
         feature_map_pair[0], feature_map_pair[1], source_mask, target_mask,
         self._train_search_size, 'cosine')
    all_corr_aepe = self._loss_module.get_average_epe(correspondence_maps,
                                                      flow_gt, source_mask)
    masked_corr_aepe = self._loss_module.get_average_epe(
        correspondence_maps, flow_gt, flow_mask)
    self.correspondence_maps = correspondence_maps

    scalar_summaries = {key: value for key, value in loss_summaries.items()}
    scalar_summaries['all_aepe_search'] = all_corr_aepe
    scalar_summaries['masked_aepe_search'] = masked_corr_aepe
    image_summaries = self._get_image_summaries(input_batch)
    training_loss = scalar_summaries['training_loss']

    return training_loss, scalar_summaries, image_summaries

  def get_eval_outputs(
      self, input_batch: Dict[str, tf.Tensor]
  ) -> Tuple[tf.Tensor, Dict[str, tf.Tensor], Dict[str, tf.Tensor]]:
    """Run the geodesic feature learning model for evaluation.

    Args:
      input_batch: A dictionary holding the data batch.

    Returns:
      training_loss: A scalar tensor.
      scalar_summaries: A dictionary holding scalar tensors to summarize.
      image_summaries: A dictionary holding image tensors to summarize.
    """
    image_pair = input_batch['images']
    feature_map_pair = self.call(image_pair)

    if 'flow_mask' not in input_batch:
      input_batch['flow_mask'] = tf.math.less(
          tf.reduce_mean(tf.abs(input_batch['flows']), axis=-1), _UINT_16)
    if 'masks' not in input_batch:
      masks = input_batch['flow_mask'][:, tf.newaxis, :, :, :]
      input_batch['masks'] = tf.tile(masks, [1, 2, 1, 1, 1])
    input_batch['flows'] = input_batch['flows'] * self._flow_scale_factor

    flow_gt = input_batch['flows']
    flow_mask = tf.cast(input_batch['flow_mask'], tf.float32)
    source_mask = input_batch['masks'][:, 0]
    target_mask = input_batch['masks'][:, 1]
    (correspondence_maps,
     _) = descriptor_utils.batch_find_correspondence_by_nearest_search(
         feature_map_pair[0], feature_map_pair[1], source_mask, target_mask,
         self._eval_search_size, 'cosine')
    all_corr_aepe = self._loss_module.get_average_epe(correspondence_maps,
                                                      flow_gt, source_mask)
    masked_corr_aepe = self._loss_module.get_average_epe(
        correspondence_maps, flow_gt, flow_mask)
    self.correspondence_maps = correspondence_maps

    scalar_summaries = {}
    scalar_summaries['all_aepe_search'] = all_corr_aepe
    scalar_summaries['masked_aepe_search'] = masked_corr_aepe
    image_summaries = self._get_image_summaries(input_batch)
    training_loss = tf.constant(0.0)

    return (training_loss, scalar_summaries, image_summaries)

  def _get_image_summaries(
      self, input_batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Get the image summaries for visualization.

    Args:
      input_batch: A dictionary holding the data batch.

    Returns:
      image_summaries: A dictionary holding image tensors to summarize.
    """
    image_summaries = {}
    images = input_batch['images']
    source_mask = tf.cast(input_batch['masks'][:, 0], tf.float32)
    image_summaries['first_image'] = (images[:, 0] + 1.0) / 2.0
    image_summaries['second_image'] = (images[:, 1] + 1.0) / 2.0
    gt_warped_image = flow_ops.bilinear_warp(
        input_batch['images'][:, 1, ...],
        input_batch['flows'] * self._flow_scale_factor) * source_mask
    image_summaries['gt_warped_image'] = (gt_warped_image + 1.0) / 2.0
    image_summaries['gt_flow'] = flow_ops.create_flow_image(
        input_batch['flows'] * self._flow_scale_factor,
        saturate_magnitude=30) / 255
    image_summaries['gt_diff_image'] = tf.abs(
        image_summaries['gt_warped_image'] - image_summaries['first_image'])
    image_summaries['flow_search'] = flow_ops.create_flow_image(
        self.correspondence_maps * source_mask, saturate_magnitude=30) / 255
    warped_image_search = flow_ops.bilinear_warp(
        input_batch['images'][:, 1, ...],
        self.correspondence_maps) * source_mask
    image_summaries['warped_image_search'] = (warped_image_search + 1.0) / 2.0
    image_summaries['diff_image_search'] = tf.abs(
        image_summaries['warped_image_search'] - image_summaries['first_image'])

    return image_summaries
