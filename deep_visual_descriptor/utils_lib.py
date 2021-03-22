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

"""Utility library for deep descriptor learning framework."""

from typing import Tuple

import tensorflow as tf

from google3.vr.perception.deepholodeck.human_correspondence.optical_flow import utils_lib
from google3.research.vision.piedpiper.brain.python.ops import flow_ops


def compute_l2_distance_matrix(
    source_samples: tf.Tensor,
    target_samples: tf.Tensor,
) -> tf.Tensor:
  """Computes L2 distance between two set of features.

  Args:
    source_samples: A tensor of size M x C where M is the number of source
      sampled feature, C is the feature dimension.
    target_samples: A tensor of size N x C where N is the number of target
      sampled feature, C is the feature dimension.

  Returns:
    A tensor of size M x N, which stores L2 distance from source sampled
      features to target sampled features.
  """

  inner_source = tf.reduce_sum(
      source_samples * source_samples, axis=1, keepdims=True)
  inner_target = tf.reduce_sum(
      target_samples * target_samples, axis=1, keepdims=True)
  correlation_source_target = tf.matmul(
      source_samples, tf.transpose(target_samples, perm=(1, 0)))
  distance_matrix = inner_source - 2 * correlation_source_target + tf.transpose(
      inner_target, perm=(1, 0))
  return distance_matrix


def compute_cosine_distance_matrix(source_samples: tf.Tensor,
                                   target_samples: tf.Tensor) -> tf.Tensor:
  """Computes cosine distance between two set of features.

  Args:
    source_samples: A tensor of size M x C where M is the number of source
      sampled feature, C is the feature dimension.
    target_samples: A tensor of size N x C where N is the number of target
      sampled feature, C is the feature dimension.

  Returns:
    A tensor of size M x N, which stores cosine distance from
      source sampled features to target sampled features.
  """
  distance_matrix = tf.matmul(source_samples,
                              tf.transpose(target_samples, perm=(1, 0)))
  return 1 - distance_matrix


def find_correspondence_by_nearest_neighbor(
    source_feature_map: tf.Tensor,
    target_feature_map: tf.Tensor,
    source_mask: tf.Tensor,
    target_mask: tf.Tensor,
    method: str = 'cosine') -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes correspondences between two feature maps by nearest neighbor search.

  Args:
    source_feature_map: A Tensor of size H x W x C where H x W is the size of
      source feature map, and C is the number of channels.
    target_feature_map: A Tensor of size H x W x C where H x W is the size of
      target feature map, and C is the number of channels.
    source_mask: A Tensor of size H x W x 1 indicating foreground pixels, where
      H x W is the size of source feature map.
    target_mask: A Tensor of size H x W x 1 indicating foreground pixels, where
      H x W is the size of target feature map.
    method: A string specifying the method to compute distance between two
      sampled feature. Could be either 'L2' or 'cosine'.

  Returns:
    A list of two Tensors:
      - A Tensor of size H x W x 2, where H x W is the size of source feature
      map. A flow field to indicate correspondence.
      - A Tensor of size H x W x 1, where H x W is the size of source feature
      map. A distance map to indicate the distance of nearest neighbor.
  Raises:
    Exception: if the method type of 'method' is not supported.
  """
  height, width, feat_dims = tf.unstack(tf.shape(source_feature_map))

  source_feature_map = tf.reshape(source_feature_map,
                                  (height * width, feat_dims))
  target_feature_map = tf.reshape(target_feature_map,
                                  (height * width, feat_dims))
  source_mask = tf.reshape(source_mask, (height * width,))
  target_mask = tf.reshape(target_mask, (height * width,))

  # Fetch feature of foreground pixels.
  masked_source_feature = source_feature_map[source_mask]
  masked_target_feature = target_feature_map[target_mask]

  # Fetch coordinates of foreground pixels.
  coords = tf.meshgrid(tf.range(height), tf.range(width), indexing='ij')
  coords = tf.cast(tf.stack(coords, axis=-1), tf.int32)
  coords = tf.reshape(coords, (height * width, 2))
  src_coords = coords[source_mask]
  tgt_coords = coords[target_mask]

  # Initialize correspondence and distance map.
  correspondence = tf.zeros((height, width, 2), dtype=tf.float32)
  distance_map = tf.zeros((height, width, 1), dtype=tf.float32)

  if method == 'cosine':
    distance_matrix = compute_cosine_distance_matrix(masked_source_feature,
                                                     masked_target_feature)
  elif method == 'L2':
    distance_matrix = compute_l2_distance_matrix(masked_source_feature,
                                                 masked_target_feature)
  else:
    raise Exception('method %s not implemented' % method)
  distances, indices = tf.nn.top_k(-distance_matrix, k=1)

  nearest_tgt_coords = tf.gather_nd(tgt_coords, indices, batch_dims=0)

  displacement = tf.cast(nearest_tgt_coords - src_coords, tf.float32)
  correspondence = tf.tensor_scatter_nd_update(correspondence, src_coords,
                                               displacement)
  correspondence = tf.reverse(correspondence, axis=[2])

  distance_map = tf.tensor_scatter_nd_update(distance_map, src_coords,
                                             -distances)

  return correspondence, distance_map


def batch_find_correspondence_by_nearest_search(src_feature_maps: tf.Tensor,
                                                tgt_feature_maps: tf.Tensor,
                                                src_masks: tf.Tensor,
                                                tgt_masks: tf.Tensor,
                                                search_size: Tuple[int,
                                                                   int] = (192,
                                                                           128),
                                                method: str = 'cosine'):
  """Computes correspondence between two batch of feature maps.

  Args:
    src_feature_maps: A Tensor of size N x H x W x C where N is the batch size,
      H x W is the size of source feature map, and C is the number of channels.
    tgt_feature_maps: A Tensor of size N x H x W x C where N is the batch size,
      H x W is the size of target feature map, and C is the number of channels.
    src_masks: A Tensor of size N x H x W x 1 indicating foreground pixels,
      where N is the batch size, H x W is the size of source feature map.
    tgt_masks: A Tensor of size N x H x W x 1 indicating foreground pixels,
      where N is the batch size, H x W is the size of target feature map.
    search_size: A Tuple of two int specifying at which resolution to compute
      correspondence
    method: A string specifying the method to compute distance between two
      sampled feature.

  Returns:
    A list of two Tensors:
      - A Tensor of size N x H x W x 2, where N is the batch size, H x W is the
      size of source feature map. A flow field to indicate correspondence.
      - A Tensor of size N x H x W x 1, where N si the batch size, H x W is the
      size of source feature map. A distance map to indicate the distance of
      nearest neighbor.

  """
  batch, height, width, _ = src_masks.get_shape().as_list()
  original_size = (height, width)

  src_feature_maps = tf.image.resize(src_feature_maps, search_size)
  tgt_feature_maps = tf.image.resize(tgt_feature_maps, search_size)

  src_masks = tf.cast(src_masks, tf.float32)
  src_masks = tf.image.resize(src_masks, search_size, 'nearest')
  src_masks = tf.cast(src_masks, tf.bool)
  tgt_masks = tf.cast(tgt_masks, tf.float32)
  tgt_masks = tf.image.resize(tgt_masks, search_size, 'nearest')
  tgt_masks = tf.cast(tgt_masks, tf.bool)

  correspondence_maps = []
  distance_maps = []

  for batch_id in range(batch):
    src_feature_map = src_feature_maps[batch_id]
    tgt_feature_map = tgt_feature_maps[batch_id]
    src_mask = src_masks[batch_id]
    tgt_mask = tgt_masks[batch_id]

    correspondence, distance_map = find_correspondence_by_nearest_neighbor(
        src_feature_map, tgt_feature_map, src_mask, tgt_mask, method)

    correspondence_maps.append(correspondence)
    distance_maps.append(distance_map)

  correspondence_maps = tf.stack(correspondence_maps, axis=0)
  distance_maps = tf.stack(distance_maps, axis=0)

  correspondence_maps = utils_lib.compute_upsample_flow(correspondence_maps,
                                                        original_size)
  distance_maps = tf.image.resize(distance_maps, original_size)

  return correspondence_maps, distance_maps


def sparse_bilinear_sample(image: tf.Tensor,
                           xy_coords: tf.Tensor,
                           height_coords: float = 1.0,
                           width_coords: float = 1.0,
                           pad_mode: str = 'EDGE'):
  """Perform bilinear sampling on `image` using the given sparse `xy_coords`.

  This function sample the pixels from `image` by given sparse pixel
  coordinates. The pixel coordinates could be defined on a image with different
  image resolution, while we can scale the value of xy coordinates to fit the
  resolution of input image.

  Args:
    image: A tensor of size B x H x W x C where B is number of the batch size, H
      x W is the size of the input image, C is the channels of the image.
    xy_coords: A tensor of size B x N x 2, where B is number of the batch size,
      N is the number of pixels to be sampled.
    height_coords: A float indicating the height of image where 'xy_coords' are
      defined.
    width_coords: A float indicating the width of image where 'xy_coords' are
      defined.
    pad_mode: The padding mode: either "EDGE" or "ZERO".

  Returns:
    A tensor of size B x N x C, where B is number of the batch size,
      N is the number of pixels to be sampled, C is the channels of the image.
  """
  _, height_image, width_image, _ = tf.unstack(tf.shape(image))

  x_coords, y_coords = tf.split(xy_coords, [1, 1], axis=-1)

  x_coords = x_coords * tf.cast(width_image, tf.float32) / width_coords
  y_coords = y_coords * tf.cast(height_image, tf.float32) / height_coords
  output = flow_ops.bilinear_sample(image, x_coords, y_coords, pad_mode)

  return output


def dense_bilinear_sample(image: tf.Tensor,
                          flow: tf.Tensor,
                          pad_mode: str = 'EDGE'):
  """Perform bilinear warping on `image` using the given `flow` field.

  This function warps `image` given pixel offsets (the `flow` field). The flow
  field is interpreted as reverse flow, i.e. for every pixel location in the new
  (warped) image, `flow` specifies the <x,y> offset for the pixel in `image`.

  Note: `image` and `flow` can have different resolution, while we will scale
  the value of 'flow'

  Args:
    image: A tensor of size B x H x W x C, representing the input image, where B
      is number of the batch size, H x W is the size of the input image.
    flow: A tensor of size B x H x W x 2, representing the horizontal and
      vertical offset for the flow, where H x W is the size of the 'flow' field.
    pad_mode: The padding mode: either "EDGE" or "ZERO".

  Returns:
    A tensor of size [batch_size, height, width, channels] holding the
      warped image.
  """
  _, height_image, width_image, _ = tf.unstack(tf.shape(image))
  _, height_flow, width_flow, _ = tf.unstack(tf.shape(flow))

  # Generates pixel range over [0..w/h-1].
  x_coords = tf.cast(tf.range(width_flow), tf.float32)
  y_coords = tf.cast(tf.range(height_flow), tf.float32)

  # Reshapes ranges to allow broadcasting.
  x_coords = tf.reshape(x_coords, [1, 1, width_flow, 1])
  y_coords = tf.reshape(y_coords, [1, height_flow, 1, 1])

  # Resamples coordinates are sum of mesh and <u, v> flow field.
  x_flow, y_flow = tf.split(flow, [1, 1], axis=-1)
  x_coords += x_flow
  y_coords += y_flow
  # Rescales xy coordinate value to fit the resolution of image.
  x_coords = x_coords * tf.cast(width_image, tf.float32) / tf.cast(
      width_flow, tf.float32)
  y_coords = y_coords * tf.cast(height_image, tf.float32) / tf.cast(
      height_flow, tf.float32)
  output = flow_ops.bilinear_sample(image, x_coords, y_coords, pad_mode)

  return output
