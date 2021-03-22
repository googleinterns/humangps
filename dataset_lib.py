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

"""Data Input functions for PWC-Net.

It loads image pairs and the ground truth flow field from the first to the
second image, applies data augmentation, and returns the augmented data

"""
from typing import Any, Dict, Text, Tuple

import tensorflow.google as tf

from google3.vr.perception.deepholodeck.human_correspondence.optical_flow import utils_lib
from google3.vr.perception.tensorflow.data import dataset_loading

# The GT flow will be divided by this scale factor (default 20.0) and used as
# supervision signal, so as to use the learning rate schedule proposed in
# the FlowNet paper
_FLOW_SCALE_FACTOR = 20.0
_FORWARD_FLOW_SCALE = 64.0
_FORWARD_FLOW_OFFSET = 32768.0
_UINT_16_FACTOR = 65535.0
_MAX_GEODESIC = 5.0
_LOW_RESOLUTION_SIZE = (384, 256)
_MASK_THRESHOLD = 127

_BRIGHTNESS_MAX_DELTA = 0.1
_CONTRAST_LOWER = 0.8
_CONTRAST_UPPER = 1.2
_HUE_MAX_DELTA = 0.02
_RATIO_SMPL_RP = (0.1, 0.9)
_RATIO_RP_HOLODECK = (0.85, 0.15)
_NUM_MAPS = 4


def augment_colors(batch):
  """Perform color augmentation."""
  images = batch['images']
  masks = batch['masks']

  # Convert to [0, 1] before applying color augmentation.
  images = (images + 1) / 2
  source_image = images[0, :, :, :]
  target_image = images[1, :, :, :]

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

  images = tf.stack([source_image, target_image])
  images = images * tf.cast(masks, dtype=tf.float32)
  batch['images'] = images * 2 - 1
  return batch


def _preprocess(example: Dict[Text, tf.Tensor],
                attributes: Tuple[Text]) -> Dict[Text, Any]:
  """Preprocess the example.

  Args:
    example: A dictionary of Text keys and Tensor values.
    attributes: A list of string to define the attributes of the data.

  Returns:
    output_example: Processed data.
  """
  output_example = dict()

  # Extract the images, and scale them to [-1, 1].
  source_image = tf.cast(example['source_rgb'], tf.float32) / 127.5 - 1.0
  target_image = tf.cast(example['target_rgb'], tf.float32) / 127.5 - 1.0
  images = tf.stack([source_image, target_image])
  _, height, width, _ = images.get_shape().as_list()
  if 'low_resolution' in attributes:
    images = tf.image.resize(images, _LOW_RESOLUTION_SIZE, method='bilinear')

  # Extract the masks.
  source_mask = example['source_mask']
  target_mask = example['target_mask']
  masks = tf.stack([source_mask, target_mask])
  if 'low_resolution' in attributes:
    masks = tf.image.resize(masks, _LOW_RESOLUTION_SIZE, method='bilinear')
  masks = masks > _MASK_THRESHOLD

  # Extract the flow, and scale it.
  flows = (tf.cast(example['forward_flow'], tf.float32) -
           _FORWARD_FLOW_OFFSET) / _FORWARD_FLOW_SCALE
  flows = flows / _FLOW_SCALE_FACTOR
  flow_mask = example['flow_mask'] > _MASK_THRESHOLD
  if 'low_resolution' in attributes:
    flows = utils_lib.compute_upsample_flow(flows, _LOW_RESOLUTION_SIZE)
    flow_mask = tf.image.resize(
        example['flow_mask'], _LOW_RESOLUTION_SIZE, method='nearest')
    flow_mask = flow_mask > _MASK_THRESHOLD

  # Extract geodesic maps
  source_geodesic_maps = tf.cast(example['source_geodesic_maps'],
                                 tf.float32) / _UINT_16_FACTOR * _MAX_GEODESIC
  source_geodesic_maps = tf.reshape(source_geodesic_maps,
                                    [height, width, _NUM_MAPS])
  target_geodesic_maps = tf.cast(example['target_geodesic_maps'],
                                 tf.float32) / _UINT_16_FACTOR * _MAX_GEODESIC
  target_geodesic_maps = tf.reshape(target_geodesic_maps,
                                    [height, width, _NUM_MAPS])
  source_geo_centers = example['source_geo_centers']
  target_geo_centers = example['target_geo_centers']
  if 'low_resolution' in attributes:
    source_geodesic_maps = tf.image.resize(
        source_geodesic_maps, _LOW_RESOLUTION_SIZE, method='nearest')
    target_geodesic_maps = tf.image.resize(
        target_geodesic_maps, _LOW_RESOLUTION_SIZE, method='nearest')
    source_geo_centers = source_geo_centers / (height / _LOW_RESOLUTION_SIZE[0])
    target_geo_centers = target_geo_centers / (height / _LOW_RESOLUTION_SIZE[0])

  # Extract cross geodesic maps
  source_cross_geodesic_maps = tf.cast(
      example['source_cross_geodesic_maps'],
      tf.float32) / _UINT_16_FACTOR * _MAX_GEODESIC
  source_cross_geodesic_maps = tf.reshape(source_cross_geodesic_maps,
                                          [height, width, _NUM_MAPS])

  target_cross_geodesic_maps = tf.cast(
      example['target_cross_geodesic_maps'],
      tf.float32) / _UINT_16_FACTOR * _MAX_GEODESIC
  target_cross_geodesic_maps = tf.reshape(target_cross_geodesic_maps,
                                          [height, width, _NUM_MAPS])
  source_cross_geo_centers = example['source_cross_geo_centers']
  target_cross_geo_centers = example['target_cross_geo_centers']
  if 'low_resolution' in attributes:
    source_cross_geodesic_maps = tf.image.resize(
        source_cross_geodesic_maps, _LOW_RESOLUTION_SIZE, method='nearest')
    target_cross_geodesic_maps = tf.image.resize(
        target_cross_geodesic_maps, _LOW_RESOLUTION_SIZE, method='nearest')
    source_cross_geo_centers = source_cross_geo_centers / (
        height / _LOW_RESOLUTION_SIZE[0])
    target_cross_geo_centers = target_cross_geo_centers / (
        height / _LOW_RESOLUTION_SIZE[0])

  # Extract geodesic matrix
  source_geodesic_matrix = tf.cast(example['source_geodesic_matrix'],
                                   tf.float32) / _UINT_16_FACTOR * _MAX_GEODESIC
  target_geodesic_matrix = tf.cast(example['target_geodesic_matrix'],
                                   tf.float32) / _UINT_16_FACTOR * _MAX_GEODESIC
  source_geo_keypoints = example['source_geo_keypoints']
  target_geo_keypoints = example['target_geo_keypoints']
  if 'low_resolution' in attributes:
    source_geo_keypoints = source_geo_keypoints / (
        height / _LOW_RESOLUTION_SIZE[0])
    target_geo_keypoints = target_geo_keypoints / (
        height / _LOW_RESOLUTION_SIZE[0])

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
  dataset_proto_path = dataset_params['dataset_proto_path']
  attributes = dataset_params['attributes']

  def parse_and_decode_fn(key, serialized_example):
    decoded_features = dataset_loading.get_parse_and_decode_func(
        *dataset_loading.parse_and_decode_info_from_proto(
            filename=dataset_proto_path))(
                serialized_example)
    decoded_features['key'] = key
    return decoded_features

  filenames_dataset = tf.data.Dataset.list_files(
      dataset_params['data_path'], shuffle=dataset_params['is_training'])
  dataset = filenames_dataset.interleave(
      lambda item: tf.data.SSTableDataset(item).map(parse_and_decode_fn),
      block_length=2,
      num_parallel_calls=tf.data.AUTOTUNE)
  dataset = dataset.map(
      lambda x: _preprocess(x, attributes), num_parallel_calls=tf.data.AUTOTUNE)

  if 'renderpeople_path' in dataset_params:
    filenames_dataset = tf.data.Dataset.list_files(
        dataset_params['renderpeople_path'],
        shuffle=dataset_params['is_training'])
    renderpeople_dataset = filenames_dataset.interleave(
        lambda item: tf.data.SSTableDataset(item).map(parse_and_decode_fn),
        block_length=2,
        num_parallel_calls=tf.data.AUTOTUNE)
    renderpeople_attributes = attributes.copy()
    renderpeople_dataset = renderpeople_dataset.map(
        lambda x: _preprocess(x, renderpeople_attributes),
        num_parallel_calls=tf.data.AUTOTUNE)
    datasets = [dataset, renderpeople_dataset]
    dataset = tf.data.experimental.sample_from_datasets(datasets,
                                                        _RATIO_SMPL_RP)

  if 'holodeck_data_path' in dataset_params:
    filenames_dataset = tf.data.Dataset.list_files(
        dataset_params['holodeck_data_path'],
        shuffle=dataset_params['is_training'])
    holodeck_dataset = filenames_dataset.interleave(
        lambda item: tf.data.SSTableDataset(item).map(parse_and_decode_fn),
        block_length=2,
        num_parallel_calls=tf.data.AUTOTUNE)
    holodeck_attributes = attributes.copy()
    holodeck_attributes.append('has_no_geodesic')
    holodeck_dataset = holodeck_dataset.map(
        lambda x: _preprocess(x, holodeck_attributes),
        num_parallel_calls=tf.data.AUTOTUNE)
    datasets = [dataset, holodeck_dataset]
    dataset = tf.data.experimental.sample_from_datasets(datasets,
                                                        _RATIO_RP_HOLODECK)

  if dataset_params['is_training']:
    dataset = dataset.map(augment_colors, num_parallel_calls=tf.data.AUTOTUNE)

  dataset = dataset.batch(dataset_params['batch_size'], drop_remainder=True)
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
