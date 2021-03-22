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

"""Loss library for Geodesic Feature Model."""
from typing import Any, Dict, Sequence

import tensorflow as tf

from google3.vr.perception.deepholodeck.human_correspondence.deep_visual_descriptor import utils_lib as descriptor_utils
from google3.vr.perception.deepholodeck.human_correspondence.optical_flow import utils_lib as flow_utils

_EPSILON = 1e-12
_M = 1.0
_TAU = 0.2
_ALPHA = 6.0
_BETA = 3.0
_GEODESIC_THRESHOLD = 0.05
_LEVEL_RANGE = 4
_SIZE_COORDS = (384, 256)


class GeodesicFeatureLoss:
  """The loss function class for learning geodesic descriptor."""

  def __init__(self, loss_hparams: Dict[str, Any]):
    """Initializes loss class for training geodesic feature descriptor.

    Args:
      loss_hparams: A Dict to specify the weight for each loss component.
    """

    self._loss_hparams = loss_hparams

    # Pre-defined hyperparameters
    # Set weight for multi-scale loss.
    self._weight = [
        1.0, 0.25, 0.125, 0.0125, 0.00625, 0.00625, 0.00625, 0.00625
    ]
    # Hyperparameters in computing threholded hinged triplet loss
    self._m = _M
    self._tau = _TAU
    # Hyperparameters to control the slope of the gradient from dense geodesic
    # loss.
    self._alpha = _ALPHA
    self._beta = _BETA
    self._geodesic_threhold = _GEODESIC_THRESHOLD

    # Define the levels of feature pyramid will be considered in computing
    # losses.
    self._level_range = _LEVEL_RANGE
    self._height_coords, self._width_coords = _SIZE_COORDS

  def set_coord_size(self, height_coords: int, width_coords: int):
    """Set 'coord_size' corresponding to the scale of sampled coordinates."""
    self._height_coords = height_coords
    self._width_coords = width_coords

  @classmethod
  def get_average_epe(cls,
                      flow_pred: tf.Tensor,
                      flow_gt: tf.Tensor,
                      flow_mask: tf.Tensor,
                      flow_scale_factor: float = 1.0) -> tf.Tensor:
    """Calculates the averaged end-point-error.

    Args:
      flow_pred: Predicted optical flow of size B x H x W x 2, where B is the
        number of batch size, H x W is the size of predicted optical flow.
      flow_gt: Ground-truth optical flow of size B x H x W x 2, where B is the
        number of batch size, H x W is the size of ground-truth optical flow.
      flow_mask: Ground-truth flow mask of size B x H x W x 1, which indicates
        the co-visible pixels.
      flow_scale_factor: The supervision signal at each pyramid level is ground
        truth/flow_scale_factor (20. by FlowNet and PWC-Net, set it to 0 for
        nearest neghbor search).

    Returns:
      Average end pont error.
    """
    size_gt = tf.shape(flow_gt)[-3:-1]
    flow_pred = flow_utils.compute_upsample_flow(flow_pred, size_gt)
    flow_mask = tf.cast(flow_mask, tf.float32)
    diff = tf.square(flow_gt - flow_pred * flow_scale_factor)
    err = tf.math.sqrt(tf.reduce_sum(diff, axis=-1, keepdims=True) + _EPSILON)
    return tf.math.divide_no_nan(
        tf.reduce_sum(err * flow_mask), tf.reduce_sum(flow_mask))

  def get_intra_consistency_loss(self,
                                 source_feature_pyramid: Sequence[tf.Tensor],
                                 target_feature_pyramid: Sequence[tf.Tensor],
                                 flow_gt: tf.Tensor,
                                 flow_mask: tf.Tensor) -> tf.Tensor:
    """Computes consistency loss for intra subjects.

    Args:
      source_feature_pyramid: A list of tensors. For each tensor, the size is
        [B, H/2^L, W/2^L, C], where L is the (level-1) of the pyramid. B is the
        number of batch size, H x W is the original image size.
      target_feature_pyramid: A list of L tensors. For each tensor, the size is
        [B, H/2^L, W/2^L, C], where L is the (level-1) of the pyramid. B is the
        number of batch size, H x W is the original image size.
      flow_gt: Ground-truth optical flow of size B x H x W x 2, where B is the
        number of batch size, H x W is the size of ground-truth optical flow.
      flow_mask: Ground-truth flow mask of size B x H x W x 1, which indicates
        the co-visible pixels.

    Returns:
      Average end pont error: A scalar tensor.
    """
    zero_flow = tf.zeros_like(flow_gt)

    losses = []
    for i, (source_feature, target_feature) in enumerate(
        zip(source_feature_pyramid, target_feature_pyramid)):
      sampled_source_feature = descriptor_utils.dense_bilinear_sample(
          source_feature, zero_flow)
      sampled_target_feature = descriptor_utils.dense_bilinear_sample(
          target_feature, flow_gt)

      feature_distance = (1 - tf.reduce_sum(
          sampled_source_feature * sampled_target_feature,
          axis=-1,
          keepdims=True))
      loss = tf.math.divide_no_nan(
          tf.reduce_sum(feature_distance * flow_mask), tf.reduce_sum(flow_mask))
      loss = loss * self._weight[i]
      losses.append(loss)
    return tf.add_n(losses)

  def compute_dense_geodesic_loss(self,
                                  source_feature_pyramid: Sequence[tf.Tensor],
                                  target_feature_pyramid: Sequence[tf.Tensor],
                                  geodesic_centers: tf.Tensor,
                                  geodesic_maps: tf.Tensor,
                                  geodesic_mask: tf.Tensor) -> tf.Tensor:
    """Computes dense geodesic loss on source and target feature pyramid.

    Args:
      source_feature_pyramid: A list of L tensors. For each tensor, the size is
        [B, H/2^L, W/2^L, C], where L is the (level-1) of the pyramid. B is the
        number of batch size, H x W is the original image size.
      target_feature_pyramid: A list of L tensors. For each tensor, the size is
        [B, H/2^L, W/2^L, C], where L is the (level-1) of the pyramid. B is the
        number of batch size, H x W is the original image size.
      geodesic_centers: A tensor of size B x N x 2, where B is the number of
        batch size, N is the number of the sampled center pixels.
      geodesic_maps: A tensor of size B x H x W x N, where B is the number of
        batch size, H x W is the original image size, N is the number of the
        sampled pixels.
      geodesic_mask: Ground-truth foreground mask of size B x H x W x 1,
        indicating foreground pixels of source image.

    Returns:
      Dense geodesic loss.
    """
    batch_size, num_center = geodesic_centers.get_shape().as_list()[:2]

    # Generate pixel range over [0..w/h-1].
    xy_coords = flow_utils.coords_grid(batch_size, self._height_coords,
                                       self._width_coords)
    geodesic_maps = tf.transpose(geodesic_maps, [0, 3, 1, 2])[..., tf.newaxis]
    geodesic_mask = geodesic_mask[:, tf.newaxis, :, :, :]

    losses = []
    for level, (source_feature, target_feature) in enumerate(
        zip(source_feature_pyramid, target_feature_pyramid)):
      center_feature = descriptor_utils.sparse_bilinear_sample(
          source_feature,
          geodesic_centers,
          height_coords=self._height_coords,
          width_coords=self._width_coords,
      )
      dense_feature_map = descriptor_utils.dense_bilinear_sample(
          target_feature,
          xy_coords,
      )

      center_feature = center_feature[:, :, tf.newaxis, tf.newaxis, :]
      dense_feature_map = dense_feature_map[:, tf.newaxis, :, :, :]
      distance = 1 - tf.reduce_sum(
          center_feature * dense_feature_map, axis=-1, keepdims=True)

      loss = tf.math.log(1 +
                         tf.math.exp((geodesic_maps - distance) * self._alpha -
                                     self._beta))
      loss = tf.math.divide_no_nan(
          tf.reduce_sum(loss * geodesic_mask),
          tf.reduce_sum(geodesic_mask * num_center))

      loss = loss * self._weight[level]
      losses.append(loss)
    return tf.add_n(losses)

  def get_dense_geodesic_loss(self, feature_pyramid: Sequence[tf.Tensor],
                              geodesic_centers: tf.Tensor,
                              geodesic_maps: tf.Tensor) -> tf.Tensor:
    """Computes dense geodesic loss from a single view.

    Args:
      feature_pyramid: A list of L tensors. For each tensor, the size is [B,
        H/2^L, W/2^L, C], where L is the (level-1) of the pyramid. B is the
        number of batch size, H x W is the original image size.
      geodesic_centers: A tensor of size B x N x 2, where B is the number of
        batch size, N is the number of the sampled center pixels.
      geodesic_maps: A tensor of size B x H x W x N, where B is the number of
        batch size, H x W is the original image size, N is the number of the
        sampled pixels.

    Returns:
      Dense geodesic loss from a single view.
    """
    geodesic_mask = tf.cast(
        tf.reduce_sum(geodesic_maps, axis=-1, keepdims=True) >
        self._geodesic_threhold, tf.float32)

    return self.compute_dense_geodesic_loss(feature_pyramid, feature_pyramid,
                                            geodesic_centers, geodesic_maps,
                                            geodesic_mask)

  def get_cross_view_dense_geodesic_loss(
      self, source_feature_pyramid: Sequence[tf.Tensor],
      target_feature_pyramid: Sequence[tf.Tensor],
      cross_geodesic_centers: tf.Tensor,
      cross_geodesic_maps: tf.Tensor) -> tf.Tensor:
    """Computes cross-view dense geodesic loss.

    Args:
      source_feature_pyramid: A list of L tensors. For each tensor, the size is
        [B, H/2^L, W/2^L, C], where L is the (level-1) of the pyramid. B is the
        number of batch size, H x W is the original image size.
      target_feature_pyramid: A list of L tensors. For each tensor, the size is
        [B, H/2^L, W/2^L, C], where L is the (level-1) of the pyramid. B is the
        number of batch size, H x W is the original image size.
      cross_geodesic_centers: A tensor of size B x N x 2, where B is the number
        of batch size, N is the number of the sampled center pixels.
      cross_geodesic_maps: A tensor of size B x H x W x N, where B is the number
        of batch size, H x W is the original image size, N is the number of the
        sampled pixels.

    Returns:
      Cross-view dense geodesic loss from cross views.
    """
    geodesic_mask = tf.cast(
        tf.reduce_sum(cross_geodesic_maps, axis=-1, keepdims=True) > 1e-12,
        tf.float32)

    return self.compute_dense_geodesic_loss(source_feature_pyramid,
                                            target_feature_pyramid,
                                            cross_geodesic_centers,
                                            cross_geodesic_maps, geodesic_mask)

  def get_geodesic_matrix_loss(self, feature_pyramid: Sequence[tf.Tensor],
                               geodesic_coords: tf.Tensor,
                               geodesic_matrix: tf.Tensor) -> tf.Tensor:
    """Computes cross-view dense geodesic loss.

    Args:
      feature_pyramid: A list of L tensors. For each tensor, the size is [B,
        H/2^L, W/2^L, C], where L is the (level-1) of the pyramid. B is the
        number of batch size, H x W is the original image size.
      geodesic_coords: A tensor of size B x N x 2, where B is the number of
        batch size, N is the number of all the sampled pixels.
      geodesic_matrix: A tensor of size [B, N/2, N/2, 1], where B is the number
        of batch size, N is the number of the sampled pixels.

    Returns:
      Sparse ordinal geodesic loss: A scalar tensor.
    """
    batch_size, num_sample = geodesic_coords.get_shape().as_list()[:2]
    half_num_sample = num_sample // 2
    target_idxs = tf.range(half_num_sample)
    source_idxs1 = tf.random.shuffle(target_idxs) + half_num_sample
    source_idxs2 = tf.random.shuffle(target_idxs) + half_num_sample
    target_idxs = tf.tile(target_idxs[tf.newaxis, ...], [batch_size, 1])
    source_idxs1 = tf.tile(source_idxs1[tf.newaxis, ...], [batch_size, 1])
    source_idxs2 = tf.tile(source_idxs2[tf.newaxis, ...], [batch_size, 1])

    target_coords = geodesic_coords[:, :half_num_sample]
    source_coords1 = tf.gather(
        geodesic_coords, source_idxs1, axis=1, batch_dims=1)
    source_coords2 = tf.gather(
        geodesic_coords, source_idxs2, axis=1, batch_dims=1)

    tgt_src_matrix_indices1 = tf.stack(
        [target_idxs, source_idxs1 - half_num_sample], axis=-1)
    tgt_src_matrix_indices2 = tf.stack(
        [target_idxs, source_idxs2 - half_num_sample], axis=-1)

    tgt_src_geodesic1 = tf.gather_nd(
        geodesic_matrix, tgt_src_matrix_indices1, batch_dims=1)
    tgt_src_geodesic2 = tf.gather_nd(
        geodesic_matrix, tgt_src_matrix_indices2, batch_dims=1)

    positive = tf.ones_like(tgt_src_geodesic1)
    negative = -positive

    relative_flags = tf.where(
        tgt_src_geodesic1 < tgt_src_geodesic2, x=positive, y=negative)

    losses = []
    for level, feature_map in enumerate(feature_pyramid):
      target_feature = descriptor_utils.sparse_bilinear_sample(
          feature_map,
          target_coords,
          height_coords=self._height_coords,
          width_coords=self._width_coords,
      )
      source_feature1 = descriptor_utils.sparse_bilinear_sample(
          feature_map,
          source_coords1,
          height_coords=self._height_coords,
          width_coords=self._width_coords,
      )
      source_feature2 = descriptor_utils.sparse_bilinear_sample(
          feature_map,
          source_coords2,
          height_coords=self._height_coords,
          width_coords=self._width_coords,
      )

      distance1 = 1 - tf.reduce_sum(
          target_feature * source_feature1, axis=-1, keepdims=True)

      distance2 = 1 - tf.reduce_sum(
          target_feature * source_feature2, axis=-1, keepdims=True)

      loss = tf.math.log(1 + tf.math.exp(
          (relative_flags * distance1 - relative_flags * distance2) *
          self._alpha - self._beta))

      mask = tf.math.logical_and(tgt_src_geodesic1 > _EPSILON,
                                 tgt_src_geodesic2 > _EPSILON)
      mask = tf.cast(mask, tf.float32)
      loss = tf.math.divide_no_nan(
          tf.reduce_sum(loss * mask), tf.reduce_sum(mask))
      loss = loss * self._weight[level]
      losses.append(loss)

    return tf.add_n(losses)

  def thresholded_hinge_embedding_loss(self, reference_feature: tf.Tensor,
                                       positive_feature: tf.Tensor,
                                       negative_feature: tf.Tensor,
                                       reference_mask) -> tf.Tensor:
    """Computes thresholded hinge embedding loss.

    Args:
      reference_feature: A tensor of size B x N x C, where B is the number of
        batch size, N is the number of sampled reference feature, C is the
        dimensions of sampled reference feature.
      positive_feature: A tensor of size B x N x C, where B is the number of
        batch size, N is the number of sampled positive feature, C is the
        dimensions of sampled positive feature.
      negative_feature: A tensor of size B x N x C, where B is the number of
        batch size, N is the number of sampled negative feature, C is the
        dimensions of sampled negative feature.
      reference_mask: A tensor of size B x N x 1, where B is the number of batch
        size, N is the number of sampled reference feature, to indicate which
        sampled reference feature will be counted in computing loss.

    Returns:
      Thresholded hinge embedding loss.
    """
    zeros = tf.zeros_like(reference_mask)
    positive_loss = tf.maximum(
        zeros,
        tf.reduce_sum(
            tf.square(reference_feature - positive_feature),
            axis=-1,
            keepdims=True) - self._tau)
    negative_loss = tf.maximum(
        zeros, self._m + self._tau - tf.reduce_sum(
            tf.square(reference_feature - negative_feature),
            axis=-1,
            keepdims=True))
    loss = positive_loss + negative_loss
    return tf.math.divide_no_nan(
        tf.reduce_sum(loss * reference_mask), tf.reduce_sum(reference_mask))

  def get_triplet_loss(self, source_feature_pyramid: Sequence[tf.Tensor],
                       target_feature_pyramid: Sequence[tf.Tensor],
                       source_geodesic_coords: tf.Tensor, flow_gt: tf.Tensor,
                       flow_mask: tf.Tensor) -> tf.Tensor:
    """Computes triplet loss between source and target feature pyramid.

    Args:
      source_feature_pyramid: A list of L tensors. For each tensor, the size is
        [B, H/2^L, W/2^L, C], where L is the (level-1) of the pyramid. B is the
        number of batch size, H x W is the original image size.
      target_feature_pyramid: A list of L tensors. For each tensor, the size is
        [B, H/2^L, W/2^L, C], where L is the (level-1) of the pyramid. B is the
        number of batch size, H x W is the original image size.
      source_geodesic_coords: A tensor of size B x N x 2, where B is the number
        of batch size, N is the number of the sampled center pixels in source
        view.
      flow_gt: Ground-truth optical flow of size B x H x W x 2, where B is the
        number of batch size, H x W is the size of ground-truth optical flow.
      flow_mask: Ground-truth flow mask of size B x H x W x 1, which indicates
        the co-visible pixels.

    Returns:
      Triplet loss.
    """
    batch_size, num_sample = source_geodesic_coords.get_shape().as_list()[:2]

    half_num_sample = num_sample // 2
    flow_mask = tf.cast(flow_mask, dtype=tf.float32)
    reference_idxs = tf.range(half_num_sample)
    negative_idxs = tf.random.shuffle(reference_idxs) + half_num_sample
    negative_idxs = tf.tile(negative_idxs[tf.newaxis, ...], [batch_size, 1])
    reference_coords = source_geodesic_coords[:, :half_num_sample]
    negative_coords = tf.gather(
        source_geodesic_coords, negative_idxs, axis=1, batch_dims=1)

    reference_mask = descriptor_utils.sparse_bilinear_sample(
        flow_mask,
        reference_coords,
        height_coords=self._height_coords,
        width_coords=self._width_coords,
    )
    reference_flow = descriptor_utils.sparse_bilinear_sample(
        flow_gt,
        reference_coords,
        height_coords=self._height_coords,
        width_coords=self._width_coords,
    )
    negative_flow = descriptor_utils.sparse_bilinear_sample(
        flow_gt,
        negative_coords,
        height_coords=self._height_coords,
        width_coords=self._width_coords,
    )

    positive_coords = reference_coords + reference_flow
    negative_coords = negative_coords + negative_flow

    losses = []
    for level, (source_feature, target_feature) in enumerate(
        zip(source_feature_pyramid, target_feature_pyramid)):

      reference_feature = descriptor_utils.sparse_bilinear_sample(
          source_feature,
          reference_coords,
          height_coords=self._height_coords,
          width_coords=self._width_coords,
      )
      positive_feature = descriptor_utils.sparse_bilinear_sample(
          target_feature,
          positive_coords,
          height_coords=self._height_coords,
          width_coords=self._width_coords,
      )
      negative_feature = descriptor_utils.sparse_bilinear_sample(
          target_feature,
          negative_coords,
          height_coords=self._height_coords,
          width_coords=self._width_coords,
      )

      loss = self.thresholded_hinge_embedding_loss(reference_feature,
                                                   positive_feature,
                                                   negative_feature,
                                                   reference_mask)
      loss = loss * self._weight[level]
      losses.append(loss)
    return tf.add_n(losses)

  def get_training_loss(
      self, source_feature_pyramid: Sequence[tf.Tensor],
      target_feature_pyramid: Sequence[tf.Tensor],
      input_batch: Dict[str, tf.Tensor]) -> Dict[str, tf.Tensor]:
    """Computes training loss for geodesic feature.

    Args:
      source_feature_pyramid: A list of L tensors. For each tensor, the size is
        [B, H/2^L, W/2^L, C], where L is the (level-1) of the pyramid. B is the
        number of batch size, H x W is the original image size.
      target_feature_pyramid: A list of L tensors. For each tensor, the size is
        [B, H/2^L, W/2^L, C], where L is the (level-1) of the pyramid. B is the
        number of batch size, H x W is the original image size.
      input_batch: A dictionary holding the data batch. The output from
        data_lib.

    Returns:
      A loss summaries of all computed losses.
    """
    source_feature_pyramid = source_feature_pyramid[:-(self._level_range +
                                                       1):-1]
    target_feature_pyramid = target_feature_pyramid[:-(self._level_range +
                                                       1):-1]
    flow_gt = input_batch['flows']
    flow_mask = tf.cast(input_batch['flow_mask'], tf.float32)
    loss_summaries = {}
    if 'correspondence_loss_weight' in self._loss_hparams:
      correspondence_loss = self.get_intra_consistency_loss(
          source_feature_pyramid, target_feature_pyramid, flow_gt, flow_mask)

      loss_summaries['correspondence_loss'] = correspondence_loss

    if 'triplet_loss_weight' in self._loss_hparams:
      source_geodesic_coords = input_batch['source_geo_keypoints']
      triplet_loss = self.get_triplet_loss(source_feature_pyramid,
                                           target_feature_pyramid,
                                           source_geodesic_coords, flow_gt,
                                           flow_mask)

      loss_summaries['triplet_loss'] = triplet_loss

    if 'dense_geodesic_loss_weight' in self._loss_hparams:
      source_geodesic_centers = input_batch['source_geo_centers']
      target_geodesic_centers = input_batch['target_geo_centers']
      source_geodesic_maps = input_batch['source_geodesic_maps']
      target_geodesic_maps = input_batch['target_geodesic_maps']

      source_dense_geodesic_loss = self.get_dense_geodesic_loss(
          source_feature_pyramid, source_geodesic_centers, source_geodesic_maps)
      target_dense_geodesic_loss = self.get_dense_geodesic_loss(
          target_feature_pyramid, target_geodesic_centers, target_geodesic_maps)

      dense_geodesic_loss = (source_dense_geodesic_loss +
                             target_dense_geodesic_loss) / 2
      dense_geodesic_loss = tf.cond(input_batch['has_no_geodesic'][0],
                                    lambda: 0.0, lambda: dense_geodesic_loss)
      loss_summaries['dense_geodesic_loss'] = dense_geodesic_loss

    if ('cross_dense_geodesic_loss_weight' in self._loss_hparams) and (
        'source_cross_geo_centers' in input_batch):
      source_cross_geodesic_centers = input_batch['source_cross_geo_centers']
      target_cross_geodesic_centers = input_batch['target_cross_geo_centers']
      source_cross_geodesic_maps = input_batch['source_cross_geodesic_maps']
      target_cross_geodesic_maps = input_batch['target_cross_geodesic_maps']

      source_cross_dense_geodesic_loss = self.get_cross_view_dense_geodesic_loss(
          source_feature_pyramid, target_feature_pyramid,
          source_cross_geodesic_centers, source_cross_geodesic_maps)
      target_cross_dense_geodesic_loss = self.get_cross_view_dense_geodesic_loss(
          target_feature_pyramid, source_feature_pyramid,
          target_cross_geodesic_centers, target_cross_geodesic_maps)

      cross_dense_geodesic_loss = (source_cross_dense_geodesic_loss +
                                   target_cross_dense_geodesic_loss) / 2
      cross_dense_geodesic_loss = tf.cond(input_batch['has_no_geodesic'],
                                          lambda: 0.0,
                                          lambda: cross_dense_geodesic_loss)
      loss_summaries['cross_dense_geodesic_loss'] = cross_dense_geodesic_loss

    if 'sparse_ordinal_geodesic_loss_weight' in self._loss_hparams:
      source_geodesic_coords = input_batch['source_geo_keypoints']
      target_geodesic_coords = input_batch['target_geo_keypoints']
      source_geodesic_matrix = input_batch['source_geodesic_matrix']
      target_geodesic_matrix = input_batch['target_geodesic_matrix']

      source_sparse_ordinal_geodesic_loss = self.get_geodesic_matrix_loss(
          source_feature_pyramid, source_geodesic_coords,
          source_geodesic_matrix)
      target_sparse_ordinal_geodesic_loss = self.get_geodesic_matrix_loss(
          target_feature_pyramid, target_geodesic_coords,
          target_geodesic_matrix)

      sparse_ordinal_geodesic_loss = (source_sparse_ordinal_geodesic_loss +
                                      target_sparse_ordinal_geodesic_loss) / 2
      sparse_ordinal_geodesic_loss = tf.cond(
          input_batch['has_no_geodesic'], lambda: 0.0,
          lambda: sparse_ordinal_geodesic_loss)
      loss_summaries[
          'sparse_ordinal_geodesic_loss'] = sparse_ordinal_geodesic_loss

    total_loss = tf.constant(0.0)
    for key, value in loss_summaries.items():
      total_loss += value * self._loss_hparams['%s_weight' % key]
    loss_summaries['training_loss'] = total_loss
    return loss_summaries
