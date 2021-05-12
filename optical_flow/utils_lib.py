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

"""Utility library for optical flow network."""
from typing import Union, Tuple, Sequence

import tensorflow as tf


def coords_grid(
    batch_size: Union[tf.Tensor, int],
    height: Union[tf.Tensor, int],
    width: Union[tf.Tensor, int],
) -> tf.Tensor:
  """Creates a coordinate grid.

  Args:
    batch_size: A 0-D Tensor (scalar), batch size of coordinate grids.
    height: A 0-D Tensor (scalar), height of coordinate grids.
    width: A 0-D Tensor (scalar), width of coordinate grids.

  Returns:
    Coordinate grid: A Tensor of size B x H x W x 2 where B is batch size, H x W
      is the size of coordinate grid.
  """
  coords = tf.meshgrid(tf.range(height), tf.range(width), indexing='ij')
  coords = tf.cast(tf.stack(coords[::-1], axis=-1), dtype=tf.float32)
  coords = tf.expand_dims(coords, axis=0)
  coords = tf.repeat(coords, batch_size, axis=0)
  return coords


def initialize_flow(image: tf.Tensor,
                    division: int = 8) -> Tuple[tf.Tensor, tf.Tensor]:
  """Creates initial coodinates of flow field before and after update.

  Args:
    image: A Tensor of size B x H x W x C, where B is batch size, H x W is the
      image size, and C is the number of channels.
    division: A integer specifying the division factor between input image and
      flow field.

  Returns:
    Coordinate of flow field before update: A Tensor of size B x H/division x
      W/division x 2, where B is batch size, H/division x W/division is the
      flow field size.
    Coordinate of flow field after update: A Tensor of size B x H/division x
      W/division x 2, where B is batch size, H/division x W/division is the
      flow field size.
  """
  batch, height, width, _ = tf.unstack(tf.shape(image))
  pre_coords = coords_grid(batch, height // division, width // division)
  post_coords = coords_grid(batch, height // division, width // division)
  return pre_coords, post_coords


def compute_upsample_flow(flow: tf.Tensor,
                          size: Union[tf.Tensor, Tuple[int, int]]) -> tf.Tensor:
  """Resizes 'flow' to 'size' while rescaling flow value."""
  # For the boundary pixels, bilinear sampling will introduce incorrect value.
  # For example, the arms occluding the torso may have different flow value.
  flow_size = tf.unstack(flow.shape)
  if len(flow_size) == 3:
    flow_height = flow_size[0]
    flow_width = flow_size[1]
  else:
    flow_height = flow_size[1]
    flow_width = flow_size[2]

  upsampled_flow = tf.image.resize(flow, size)
  upsampled_x = upsampled_flow[..., 0] * tf.cast(
      size[1], dtype=tf.float32) / tf.cast(
          flow_width, dtype=tf.float32)
  upsampled_y = upsampled_flow[..., 1] * tf.cast(
      size[0], dtype=tf.float32) / tf.cast(
          flow_height, dtype=tf.float32)
  return tf.stack((upsampled_x, upsampled_y), axis=-1)


def create_path_drop_masks(p_flow: float,
                           p_surface: float) -> Tuple[tf.Tensor, tf.Tensor]:
  """Determines global path drop decision based on given probabilities.

  Args:
    p_flow: A scalar of float32, probability of keeping flow feature branch
    p_surface: A scalar of float32, probability of keeping surface feature
      branch

  Returns:
    final_flow_mask: A constant tensor mask containing either one or zero
      depending on the final coin flip probability.
    final_surface_mask: A constant tensor mask containing either one or
      zero depending on the final coin flip probability.
  """

  # The logic works as follows:
  # We have flipped 3 coins, first determines the chance of keeping
  # the flow branch, second determines keeping surface branch, the third
  # makes the final decision in the case where both branches were killed
  # off, otherwise the initial flow and surface chances are kept.

  random_values = tf.random.uniform(shape=[3], minval=0.0, maxval=1.0)

  keep_branch = tf.constant(1.0)
  kill_branch = tf.constant(0.0)

  flow_chances = tf.case(
      [(tf.math.less_equal(random_values[0], p_flow), lambda: keep_branch)],
      default=lambda: kill_branch)

  surface_chances = tf.case(
      [(tf.math.less_equal(random_values[1], p_surface), lambda: keep_branch)],
      default=lambda: kill_branch)

  # Decision to determine whether both branches were killed off
  third_flip = tf.math.logical_or(
      tf.cast(flow_chances, dtype=tf.bool),
      tf.cast(surface_chances, dtype=tf.bool))
  third_flip = tf.cast(third_flip, dtype=tf.float32)

  # Make a second choice, for the third case
  # Here we use a 50/50 chance to keep either flow or surface
  # If its greater than 0.5, keep the image
  flow_second_flip = tf.case(
      [(tf.math.greater(random_values[2], 0.5), lambda: keep_branch)],
      default=lambda: kill_branch)
  # If its less than or equal to 0.5, keep surface
  surface_second_flip = tf.case(
      [(tf.math.less_equal(random_values[2], 0.5), lambda: keep_branch)],
      default=lambda: kill_branch)

  final_flow_mask = tf.case([(tf.equal(third_flip, 1), lambda: flow_chances)],
                            default=lambda: flow_second_flip)

  final_surface_mask = tf.case(
      [(tf.equal(third_flip, 1), lambda: surface_chances)],
      default=lambda: surface_second_flip)

  return final_flow_mask, final_surface_mask


def build_pyramid(image: tf.Tensor,
                  num_levels: int = 7,
                  resize_method: str = 'bilinear',
                  rescale_flow: bool = False) -> Sequence[tf.Tensor]:
  """Return list of downscaled images (level 0 = original)."""
  pyramid = [image]

  size = tf.shape(image)[-3:-1]
  for _ in range(1, num_levels):
    size = size // 2
    image = tf.image.resize(pyramid[-1], size, method=resize_method)

    if rescale_flow:
      image /= 2  # half-resolution => half-size flow
    pyramid.append(image)
  return pyramid
