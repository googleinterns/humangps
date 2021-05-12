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

"""Data Input functions.

It loads image pairs and the ground truth flow field from the first to the
second image, applies data augmentation, and returns the augmented data

"""
from typing import Any, Dict, Text, Tuple
from PIL import Image

import os
import numpy as np
import tensorflow as tf

from optical_flow import utils_lib

# The GT flow will be divided by this scale factor (default 20.0) and used as
# supervision signal, so as to use the learning rate schedule proposed in
# the FlowNet paper
_FLOW_SCALE_FACTOR = 20.0
_FORWARD_FLOW_SCALE = 64.0
_FORWARD_FLOW_OFFSET = 32768.0
_UINT_16_FACTOR = 65535.0
_MAX_GEODESIC = 5.0
_LOW_RESOLUTION_SIZE = (384, 256)
_MASK_THRESHOLD = 0.5

_BRIGHTNESS_MAX_DELTA = 0.1
_CONTRAST_LOWER = 0.8
_CONTRAST_UPPER = 1.2
_HUE_MAX_DELTA = 0.02
_RATIO_ADD_RP = (0.05, 0.95)
_RATIO_ADD_HOLODECK = (0.85, 0.15)
_RATIO_ADD_SMPL_INTER = (0.95, 0.05)

_NUM_MAPS = 4

_DATA_IMG_SHAPE = (768, 512, 3)
_DATA_MASK_SHAPE = (768, 512, 1)
_DATA_FLOW_SHAPE = (768, 512, 2)
_DATA_FLOW_MASK_SHAPE = (768, 512, 1)
_DATA_KEYPOINT_SHAPE = (256, 2)
_DATA_MATRIX_SHAPE = (128, 128, 1)
_DATA_CENTER_SHAPE = (4, 2)
_DATA_MAP_SHAPE = (768, 512, 4)


def augment_colors(batch):
  """Perform color augmentation."""
  images = batch['images']
  masks = batch['masks']

  # Convert to [0, 1] before applying color augmentation.
  images = (images + 1) / 2
  source_image = images[:, 0, :, :, :]
  target_image = images[:, 1, :, :, :]

  # Adjust brightness of the source and target image respectively.
  brightness_delta = tf.random.uniform([], -_BRIGHTNESS_MAX_DELTA,
                                       _BRIGHTNESS_MAX_DELTA)
  source_image = tf.image.adjust_brightness(source_image, brightness_delta)
  brightness_delta = tf.random.uniform([], -_BRIGHTNESS_MAX_DELTA,
                                       _BRIGHTNESS_MAX_DELTA)
  target_image = tf.image.adjust_brightness(target_image, brightness_delta)

  # Adjust contrast of the source and target image respectively.
  contrast_factor = tf.random.uniform([], _CONTRAST_LOWER, _CONTRAST_UPPER)
  source_image = tf.image.adjust_contrast(source_image, contrast_factor)
  contrast_factor = tf.random.uniform([], _CONTRAST_LOWER, _CONTRAST_UPPER)
  target_image = tf.image.adjust_contrast(target_image, contrast_factor)

  # Adjust hue of the source and target image respectively.
  source_image = tf.image.random_hue(source_image, _HUE_MAX_DELTA)
  target_image = tf.image.random_hue(target_image, _HUE_MAX_DELTA)

  images = tf.stack([source_image, target_image], axis=1)
  images = images * tf.cast(masks, dtype=tf.float32)
  batch['images'] = images * 2 - 1
  return batch



def _parse_function(filename: tf.Tensor, base_path: tf.Tensor):
  """load color image, ground-truth optical flow and geodesic labels."""
  filename = filename.numpy().decode("utf-8")
  base_path = base_path.numpy().decode("utf-8")

  (src_img_filename, tgt_img_filename, src_mask_filename, tgt_mask_filename,
   flow_filename, geodesic_gt_filename) = filename.split(' ')

  src_img_filename = os.path.join(base_path, src_img_filename)
  tgt_img_filename = os.path.join(base_path, tgt_img_filename)
  src_mask_filename = os.path.join(base_path, src_mask_filename)
  tgt_mask_filename = os.path.join(base_path, tgt_mask_filename)
  flow_filename = os.path.join(base_path, flow_filename)
  geodesic_gt_filename = os.path.join(base_path, geodesic_gt_filename)

  # Read color images.
  src_img = np.array(Image.open(open(src_img_filename, 'rb')))
  tgt_img = np.array(Image.open(open(tgt_img_filename, 'rb')))

  # Read masks.
  with open(src_mask_filename, 'rb') as f:
    with np.load(f) as data:
      src_mask = data['mask']
  with open(tgt_mask_filename, 'rb') as f:
    with np.load(f) as data:
      tgt_mask = data['mask']

  # Read flows.
  with open(flow_filename, 'rb') as f:
    with np.load(f) as data:
      flows = data['flows']
      flow_mask = data['flow_mask']

  # Read geodesic groundtruth.
  with open(geodesic_gt_filename, 'rb') as f:
    with np.load(f) as data:
      source_geo_keypoints = data['source_geo_keypoints']
      source_geodesic_matrix = data['source_geodesic_matrix']
      target_geo_keypoints = data['target_geo_keypoints']
      target_geodesic_matrix = data['target_geodesic_matrix']
      source_geo_centers = data['source_geo_centers']
      source_geodesic_maps = data['source_geodesic_maps']
      target_geo_centers = data['target_geo_centers']
      target_geodesic_maps = data['target_geodesic_maps']
      source_cross_geo_centers = data['source_cross_geo_centers']
      source_cross_geodesic_maps = data['source_cross_geodesic_maps']
      target_cross_geo_centers = data['target_cross_geo_centers']
      target_cross_geodesic_maps = data['target_cross_geodesic_maps']

  data = (src_img, tgt_img, src_mask, tgt_mask, flows, flow_mask,
          source_geo_keypoints, source_geodesic_matrix, target_geo_keypoints,
          target_geodesic_matrix, source_geo_centers, source_geodesic_maps,
          target_geo_centers, target_geodesic_maps, source_cross_geo_centers,
          source_cross_geodesic_maps, target_cross_geo_centers,
          target_cross_geodesic_maps)
  return data


def tf_parse_function(filename: tf.Tensor, base_path: str, attributes: Tuple[str]):
  """Preprocess the example.

  Args:
    filename: A line of filenames indicating the relative path to data.
    attributes: A list of string to define the attributes of the data.

  Returns:
    output_example: Processed data.
  """
  (src_img, tgt_img, src_mask, tgt_mask, flows, flow_mask, source_geo_keypoints,
   source_geodesic_matrix, target_geo_keypoints, target_geodesic_matrix,
   source_geo_centers, source_geodesic_maps, target_geo_centers,
   target_geodesic_maps, source_cross_geo_centers, source_cross_geodesic_maps,
   target_cross_geo_centers, target_cross_geodesic_maps) = tf.py_function(
       _parse_function,
       inp=[filename, base_path],
       Tout=[
           tf.uint8, tf.uint8, tf.bool, tf.bool, tf.float32, tf.bool,
           tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
           tf.float32, tf.float32, tf.float32, tf.float32, tf.float32,
           tf.float32, tf.float32
       ])

  output_example = dict()

  # Extract the images, and scale them to [-1, 1].
  source_image = tf.cast(src_img, tf.float32) / 127.5 - 1.0
  target_image = tf.cast(tgt_img, tf.float32) / 127.5 - 1.0
  source_image.set_shape(_DATA_IMG_SHAPE)
  target_image.set_shape(_DATA_IMG_SHAPE)
  images = tf.stack([source_image, target_image])
  _, height, width, _ = images.get_shape().as_list()
  if 'low_resolution' in attributes:
    images = tf.image.resize(images, _LOW_RESOLUTION_SIZE, method='bilinear')

  # Extract the masks.
  source_mask = src_mask
  target_mask = tgt_mask
  source_mask.set_shape(_DATA_MASK_SHAPE)
  target_mask.set_shape(_DATA_MASK_SHAPE)
  masks = tf.stack([source_mask, target_mask])
  if 'low_resolution' in attributes:
    masks = tf.cast(masks, tf.float32)
    masks = tf.image.resize(masks, _LOW_RESOLUTION_SIZE, method='nearest')
    masks = masks > _MASK_THRESHOLD

  # Extract the flow, and scale it.
  flows = flows
  flow_mask = flow_mask
  flows.set_shape(_DATA_FLOW_SHAPE)
  flow_mask.set_shape(_DATA_FLOW_MASK_SHAPE)
  if 'low_resolution' in attributes:
    flows = utils_lib.compute_upsample_flow(flows, _LOW_RESOLUTION_SIZE)
    flow_mask = tf.cast(flow_mask, tf.float32)
    flow_mask = tf.image.resize(flow_mask, _LOW_RESOLUTION_SIZE, method='nearest')
    flow_mask = flow_mask > _MASK_THRESHOLD

  # Extract geodesic maps
  source_geodesic_maps = source_geodesic_maps
  target_geodesic_maps = target_geodesic_maps
  source_geo_centers = source_geo_centers
  target_geo_centers = target_geo_centers
  source_geodesic_maps.set_shape(_DATA_MAP_SHAPE)
  target_geodesic_maps.set_shape(_DATA_MAP_SHAPE)
  source_geo_centers.set_shape(_DATA_CENTER_SHAPE)
  target_geo_centers.set_shape(_DATA_CENTER_SHAPE)
  if 'low_resolution' in attributes:
    source_geodesic_maps = tf.image.resize(
      source_geodesic_maps, _LOW_RESOLUTION_SIZE, method='nearest')
    target_geodesic_maps = tf.image.resize(
      target_geodesic_maps, _LOW_RESOLUTION_SIZE, method='nearest')
    source_geo_centers = source_geo_centers / height * _LOW_RESOLUTION_SIZE[0]
    target_geo_centers = target_geo_centers / height * _LOW_RESOLUTION_SIZE[0]

  # Extract cross geodesic maps
  source_cross_geodesic_maps = source_cross_geodesic_maps
  target_cross_geodesic_maps = target_cross_geodesic_maps
  source_cross_geo_centers = source_cross_geo_centers
  target_cross_geo_centers = target_cross_geo_centers
  source_cross_geodesic_maps.set_shape(_DATA_MAP_SHAPE)
  target_cross_geodesic_maps.set_shape(_DATA_MAP_SHAPE)
  source_cross_geo_centers.set_shape(_DATA_CENTER_SHAPE)
  target_cross_geo_centers.set_shape(_DATA_CENTER_SHAPE)
  if 'low_resolution' in attributes:
    source_cross_geodesic_maps = tf.image.resize(
      source_cross_geodesic_maps, _LOW_RESOLUTION_SIZE, method='nearest')
    target_cross_geodesic_maps = tf.image.resize(
      target_cross_geodesic_maps, _LOW_RESOLUTION_SIZE, method='nearest')
    source_cross_geo_centers = source_cross_geo_centers / height * _LOW_RESOLUTION_SIZE[
      0]
    target_cross_geo_centers = target_cross_geo_centers / height * _LOW_RESOLUTION_SIZE[
      0]

  # Extract geodesic matrix
  source_geodesic_matrix = source_geodesic_matrix
  target_geodesic_matrix = target_geodesic_matrix
  source_geo_keypoints = source_geo_keypoints
  target_geo_keypoints = target_geo_keypoints
  source_geodesic_matrix.set_shape(_DATA_MATRIX_SHAPE)
  target_geodesic_matrix.set_shape(_DATA_MATRIX_SHAPE)
  source_geo_keypoints.set_shape(_DATA_KEYPOINT_SHAPE)
  target_geo_keypoints.set_shape(_DATA_KEYPOINT_SHAPE)
  if 'low_resolution' in attributes:
    source_geo_keypoints = source_geo_keypoints / height * _LOW_RESOLUTION_SIZE[
      0]
    target_geo_keypoints = target_geo_keypoints / height * _LOW_RESOLUTION_SIZE[
      0]

  output_example['images'] = images
  output_example['flows'] = flows
  output_example['flow_mask'] = flow_mask
  output_example['masks'] = masks

  # Add single-view geodesic matrix
  output_example['source_geo_keypoints'] = source_geo_keypoints
  output_example['source_geodesic_matrix'] = source_geodesic_matrix
  output_example['target_geo_keypoints'] = target_geo_keypoints
  output_example['target_geodesic_matrix'] = target_geodesic_matrix

  # Add single-view geodesic maps
  output_example['source_geo_centers'] = source_geo_centers
  output_example['source_geodesic_maps'] = source_geodesic_maps
  output_example['target_geo_centers'] = target_geo_centers
  output_example['target_geodesic_maps'] = target_geodesic_maps

  # Add cross-view geodesic maps
  output_example['source_cross_geo_centers'] = source_cross_geo_centers
  output_example['source_cross_geodesic_maps'] = source_cross_geodesic_maps
  output_example['target_cross_geo_centers'] = target_cross_geo_centers
  output_example['target_cross_geodesic_maps'] = target_cross_geodesic_maps

  if 'has_no_geodesic' in attributes:
    output_example['has_no_geodesic'] = True
  else:
    output_example['has_no_geodesic'] = False

  return output_example


def human_correspondence_dataset(
  dataset_params: Dict[Text, Any]) -> tf.data.Dataset:
  """Load dense human correspondence dataset.

  Args:
    dataset_params: The parameters to define the datasets, contains the dataset
      name, data path, attributes, proto path and other parameters.

  Returns:
    dataset: tf.data.Dataset to train the model for dense human correspondence.
  """
  attributes = dataset_params['attributes']

  def read_list(data_dir: str, filename_list: str = 'filename_list.txt'):
    filename_list_path = os.path.join(data_dir, filename_list)
    fp = open(filename_list_path, 'r')
    line = fp.readline()

    filenames = []
    while line:
      filename = line.replace('\n', '')
      filenames.append(filename)
      line = fp.readline()
    fp.close()
    return filenames

  if 'data_path' in dataset_params:
    filenames = read_list(dataset_params['data_path'])
    filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if dataset_params['is_training']:
      filenames_dataset = filenames_dataset.shuffle(
          len(filenames), reshuffle_each_iteration=True)
    dataset = filenames_dataset.map(
        lambda x: tf_parse_function(x, dataset_params['data_path'], attributes),
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(dataset_params['batch_size'], drop_remainder=True)

  if 'smpl_intra_path' in dataset_params:
    filenames = read_list(dataset_params['smpl_intra_path'])
    filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if dataset_params['is_training']:
      filenames_dataset = filenames_dataset.shuffle(
          len(filenames), reshuffle_each_iteration=True)
    dataset = filenames_dataset.map(
        lambda x: tf_parse_function(x, attributes),
        num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.batch(dataset_params['batch_size'], drop_remainder=True)

  if 'renderpeople_path' in dataset_params:
    filenames = read_list(dataset_params['renderpeople_path'])
    filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if dataset_params['is_training']:
      filenames_dataset = filenames_dataset.shuffle(
          len(filenames), reshuffle_each_iteration=True)
    renderpeople_dataset = filenames_dataset.map(
        lambda x: tf_parse_function(x, attributes),
        num_parallel_calls=tf.data.AUTOTUNE)
    renderpeople_dataset = renderpeople_dataset.batch(dataset_params['batch_size'], drop_remainder=True)
    datasets = [dataset, renderpeople_dataset]
    dataset = tf.data.experimental.sample_from_datasets(datasets, _RATIO_ADD_RP)

  if 'holodeck_path' in dataset_params:
    filenames = read_list(dataset_params['holodeck_path'])
    filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if dataset_params['is_training']:
      filenames_dataset = filenames_dataset.shuffle(
          len(filenames), reshuffle_each_iteration=True)
    holodeck_dataset = filenames_dataset.map(
        lambda x: tf_parse_function(x, attributes),
        num_parallel_calls=tf.data.AUTOTUNE)
    holodeck_dataset = holodeck_dataset.batch(dataset_params['batch_size'], drop_remainder=True)
    datasets = [dataset, holodeck_dataset]
    dataset = tf.data.experimental.sample_from_datasets(datasets, _RATIO_ADD_RP)

  if 'smpl_inter_path' in dataset_params:
    filenames = read_list(dataset_params['smpl_inter_path'])
    filenames_dataset = tf.data.Dataset.from_tensor_slices(filenames)
    if dataset_params['is_training']:
      filenames_dataset = filenames_dataset.shuffle(
          len(filenames), reshuffle_each_iteration=True)
    smpl_inter_dataset = filenames_dataset.map(
        lambda x: tf_parse_function(x, attributes),
        num_parallel_calls=tf.data.AUTOTUNE)
    smpl_inter_dataset = smpl_inter_dataset.batch(dataset_params['batch_size'], drop_remainder=True)
    datasets = [dataset, smpl_inter_dataset]
    dataset = tf.data.experimental.sample_from_datasets(datasets, _RATIO_ADD_RP)

  if dataset_params['is_training']:
    dataset = dataset.map(augment_colors, num_parallel_calls=tf.data.AUTOTUNE)

  dataset = dataset.prefetch(tf.data.AUTOTUNE)

  if dataset_params['is_training']:
    # Set deterministic to false for training.
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_distribute.auto_shard_policy = tf.data.experimental.AutoShardPolicy.DATA
    dataset = dataset.with_options(options)

  return dataset


def load_dataset(dataset_params: Dict[Text, Any]) -> tf.data.Dataset:
  """Load datasets.

  Args:
    dataset_params: The parameters to define the datasets, contains the dataset
      type, dataset name, data path, attributes, proto path and other
      parameters.

  Returns:
    dataset: tf.data.Dataset to train the model for dense human correspondence.
  Raises:
    Exception: if the dataset type of 'dataset' is not supported.
  """

  if dataset_params['dataset'] == 'human_correspondence_dataset':
    dataset = human_correspondence_dataset(dataset_params)
  else:
    raise Exception('Dataset: %s. is not provided' % dataset_params['dataset'])

  return dataset

