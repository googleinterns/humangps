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

# Lint as: python3
"""A pipeline to create SSTables of tf.Examples from Holodeck data.

See run_to_tf_example_pipeline.sh for example usage.
"""
import math
import os
import random
import sys
from typing import Iterable, Text, Tuple, Union

from absl import app
from absl import flags
from absl import logging
import apache_beam as beam
import numpy as np
import tensorflow.compat.v2 as tf

from google3.pipeline.flume.py import runner
from google3.pipeline.flume.py.io import sstableio
from google3.pyglib import gfile
from google3.util.task.python import error
from google3.vr.perception.deep_relighting.data_generation.python import io_utils
from google3.vr.perception.deepholodeck.hololight.data import hololight_data_utils
from google3.vr.perception.deepholodeck.human_correspondence.data.utils import data_utils as human_data_utils
from google3.vr.perception.stickercap.python import holodeck_output_proto_api

flags.DEFINE_string('output', '/tmp/output.sst', 'Output SSTable path.')
flags.DEFINE_integer('shards', 10000, 'Number of shards.')

flags.DEFINE_string(
    'holodeck_data_base_dir', None,
    'The base directory where holodeck data subject folders are present.')
flags.DEFINE_list('sequence_ids', None, 'Pattern to search capture folders.')
flags.DEFINE_list('cameras', None, 'Camera IDs to sample virtual cameras.')

flags.DEFINE_integer('image_height', 2048, 'Image height.')
flags.DEFINE_integer('image_width', 1504, 'Image width.')

flags.DEFINE_float('fov_range', 30.0,
                   'Half of field of view to sample source camera.')
flags.DEFINE_multi_float('zoom_range', [1.0, 1.0],
                         'Percentage of radius to sample source camera.')
flags.DEFINE_float('delta_fov_range', 10.0,
                   'Half of field of view to sample target camera.')
flags.DEFINE_multi_float('delta_zoom_range', [1.0, 1.0],
                         'Percentage of radius to sample target camera.')
flags.DEFINE_integer('num_sample', 100, 'Number of virtual camera to sample.')
flags.DEFINE_float('anchor_range', 0.1, 'Range to shift source anchor point.')
flags.DEFINE_float('delta_anchor_range', 0.1,
                   'Range to shift target anchor point')
flags.DEFINE_float('tilt_range', 0.0, 'Range to tilt the camera.')


FLAGS = flags.FLAGS


class ProtoPathToTFExampleFn(beam.DoFn):
  """Beam DoFn that generates a tf.train.Example."""

  def __init__(self, image_height: int, image_width: int, fov_range: float,
               zoom_range: Union[float, Tuple[float,
                                              float]], delta_fov_range: float,
               delta_zoom_range: Union[float,
                                       Tuple[float,
                                             float]], anchor_range: float,
               delta_anchor_range: float, tilt_range: float):
    """Initializes the DoFn.

    Args:
      image_height: The height of the images to be added.
      image_width: The width of the images to be added.
      fov_range: The half of the field of view to sample source cameras.
      zoom_range: The range of scale factor to sample source cameras.
      delta_fov_range: The half of the field of view to sample target cameras.
      delta_zoom_range: The range of scale factor to sample target cameras.
      anchor_range: The range to randomly shift the source anchor point.
      delta_anchor_range: The range to randomly shift the target anchor point
        from the source camera.
      tilt_range: The max degree to tilt the camera around the camera principal
        axis, usually referred as z axis.
    """
    self._image_height = image_height
    self._image_width = image_width
    self._fov_range = fov_range
    self._zoom_range = zoom_range
    self._delta_fov_range = delta_fov_range
    self._delta_zoom_range = delta_zoom_range
    self._anchor_range = anchor_range
    self._delta_anchor_range = delta_anchor_range
    self._tilt_range = tilt_range

  def process(
      self, proto_path_and_camera: Tuple[Text, Text]
  ) -> Iterable[Tuple[Text, tf.train.Example]]:
    """Computes and yields a (key, tf.Example) tuple from a path and camera.

    Args:
      proto_path_and_camera: A tuple containing the path to the holodeck output
        proto, and the camera name for which data is to be generated.

    Yields:
      A tuple of the SSTable key name and the tf.Example.
    """
    io_utils.set_swiftshader_library_path()
    proto_path, camera_name = proto_path_and_camera
    logging.info(proto_path)
    logging.info(camera_name)
    try:
      # Reads the holodeck output proto.
      output_proto = holodeck_output_proto_api.parse_holodeck_output_proto(
          proto_path)

      # Checks if camera name is present in the proto.
      proto_camera_names = holodeck_output_proto_api.get_camera_names(
          output_proto)
      if camera_name not in proto_camera_names:
        raise ValueError('Camera %s not found for proto %s.' %
                         (camera_name, proto_path))

      # Get texture map.
      uv_texture = holodeck_output_proto_api.get_blended_texture_map(
          output_proto)

      # Generate randomly sampled cameras.
      reference_camera = holodeck_output_proto_api.get_camera(
          output_proto, camera_name)
      reference_camera = hololight_data_utils.get_processed_camera(
          reference_camera, self._image_height, self._image_width)

      # Use the middle of left and right hip as the anchor point.
      keypoints = holodeck_output_proto_api.get_keypoints(output_proto)
      left_hip = np.array(keypoints['left_hip'])
      right_hip = np.array(keypoints['right_hip'])
      anchor_point = (left_hip + right_hip) / 2.0

      # Move camera a bit further to capture full body.
      reference_center = np.array(reference_camera.GetPosition())
      distance = np.linalg.norm(reference_center - anchor_point, ord=2) * 1.4
      source_degree = np.random.uniform(0, 2 * math.pi)
      x_position = np.sin(source_degree) * distance
      z_position = np.cos(source_degree) * distance
      source_camera_position = np.array([x_position, 0.1, z_position
                                        ]) + anchor_point
      reference_camera.SetPosition(source_camera_position)

      delta_fov_range = (self._delta_fov_range / 180 * math.pi)
      delta_fov_range = np.random.uniform(-delta_fov_range, delta_fov_range)
      x_position = np.sin(source_degree + delta_fov_range) * distance
      z_position = np.cos(source_degree + delta_fov_range) * distance
      target_camera_position = np.array([x_position, 0.1, z_position
                                        ]) + anchor_point
      anchor_change = np.random.uniform(-self._anchor_range, self._anchor_range,
                                        3)
      anchor_change[0] = np.random.uniform(-self._anchor_range,
                                           self._anchor_range) / 3.0
      source_anchor = anchor_point + anchor_change
      anchor_change = np.random.uniform(-self._anchor_range, self._anchor_range,
                                        3)
      anchor_change[0] = np.random.uniform(-self._anchor_range,
                                           self._anchor_range) / 3.0
      target_anchor = source_anchor + anchor_change
      source_camera = hololight_data_utils.sample_random_camera(
          reference_camera, self._fov_range, self._zoom_range, source_anchor,
          self._tilt_range)
      reference_camera.SetPosition(target_camera_position)
      target_camera = hololight_data_utils.sample_random_camera(
          reference_camera, self._fov_range, self._delta_zoom_range,
          target_anchor, self._tilt_range)

      # Calculate warping fields.
      warp_field_uv2cam = holodeck_output_proto_api.render_warp_fields(
          output_proto.mesh, [source_camera, target_camera])

      warp_field_source_uv2cam = warp_field_uv2cam[0]
      warp_field_target_uv2cam = warp_field_uv2cam[1]

      # Get parameterized mesh and convert it to SMPL format.
      parameterized_mesh = holodeck_output_proto_api.get_parameterized_mesh(
          output_proto)
      mesh = human_data_utils.add_texture_coordinate_face_to_mesh(
          parameterized_mesh)

      # Add data to record.
      feature = {}

      # Add RGB rendering.
      human_data_utils.add_rgb_rendering_holodeck(
          feature, uv_texture, warp_field_source_uv2cam[..., 0:2], mesh,
          source_camera, 'source')
      human_data_utils.add_rgb_rendering_holodeck(
          feature, uv_texture, warp_field_target_uv2cam[..., 0:2], mesh,
          target_camera, 'target')

      # Add mask
      source_mask = human_data_utils.add_foregroung_mask(
          feature, mesh, source_camera, 'source')
      _ = human_data_utils.add_foregroung_mask(feature, mesh, target_camera,
                                               'target')

      # Add optical flow.
      flow_mask = human_data_utils.add_optical_flow_rendering(
          feature, mesh, mesh, source_camera, target_camera, 'source')
      if np.sum(flow_mask) / np.sum(source_mask) < 0.5:
        return

      # Add geodesic maps
      human_data_utils.add_geodesic_maps_rendering(feature, mesh, source_camera,
                                                   'source')
      human_data_utils.add_geodesic_maps_rendering(feature, mesh, target_camera,
                                                   'target')

      # Add cross geodesic maps
      human_data_utils.add_cross_geodesic_maps_rendering(
          feature, mesh, mesh, source_camera, target_camera, 'source')
      human_data_utils.add_cross_geodesic_maps_rendering(
          feature, mesh, mesh, target_camera, source_camera, 'target')

      # Add geodesic matrix
      human_data_utils.add_geodesic_matrix_rendering(feature, mesh,
                                                     source_camera, 'source')
      human_data_utils.add_geodesic_matrix_rendering(feature, mesh,
                                                     target_camera, 'target')

      # Computes TF.Example.
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      key = str(random.randint(0, int(1e9))) + ',' + proto_path
      yield (key, example)
    except (ValueError, IOError, RuntimeError, error.StatusNotOk) as e:
      # These logging messages will show up in the logs of the worker tasks.
      logging.warning('Element %s skipped due to exception %s', proto_path,
                      str(e))
      exc_type, _, exc_tb = sys.exc_info()
      if exc_tb is not None:
        fname = os.path.split(exc_tb.tb_frame.f_code.co_filename)[1]
        logging.warning(exc_type)
        logging.warning(fname)
        logging.warning(exc_tb.tb_lineno)


def _pipeline(root):
  """Beam-on-Flume pipeline.

  Builds a beam pipeline that reads from an SSTable of olat_data protos, and
  creates and writes tf.Examples to an SSTable.

  Args:
    root: A beam.pvalue.PBegin object, the pipeline begin marker.
  """

  proto_paths = []
  for sequence_id in FLAGS.sequence_ids:
    proto_paths += gfile.Glob(
        os.path.join(FLAGS.holodeck_data_base_dir, sequence_id))
  proto_paths = [
      '%s/holodeck_output/holodeck_output_frame_0001.pb' % x
      for x in proto_paths
  ]
  print('proto_num: ', len(proto_paths))
  print('camera_num: ', len(FLAGS.cameras))

  proto_paths_and_cameras = []
  for _ in range(FLAGS.num_sample):
    for proto in proto_paths:
      for camera in FLAGS.cameras:
        proto_paths_and_cameras.append((proto, camera))

  logging.info('%d data to process in total.', len(proto_paths_and_cameras))

  proto_paths_and_cameras_pcol = root | 'create_input' >> beam.Create(
      proto_paths_and_cameras)

  print(FLAGS.zoom_range)

  key_and_tf_example = (
      proto_paths_and_cameras_pcol
      | 'convert_to_tf_example' >> beam.ParDo(
          ProtoPathToTFExampleFn(
              image_height=FLAGS.image_height,
              image_width=FLAGS.image_width,
              fov_range=FLAGS.fov_range,
              zoom_range=FLAGS.zoom_range,
              delta_fov_range=FLAGS.delta_fov_range,
              delta_zoom_range=FLAGS.delta_zoom_range,
              anchor_range=FLAGS.anchor_range,
              delta_anchor_range=FLAGS.delta_anchor_range,
              tilt_range=FLAGS.tilt_range)))

  # Writes the tf.Examples to the output SSTable.
  _ = (  # the _ keeps pylint quiet
      key_and_tf_example
      | 'WriteSSTable' >> sstableio.WriteToSSTable(
          FLAGS.output,
          key_coder=beam.coders.StrUtf8Coder(),
          value_coder=beam.coders.ProtoCoder(tf.train.Example),
          num_shards=FLAGS.shards))


def main(argv):
  del argv  # Unused.

  runner.FlumeRunner().run(_pipeline)


if __name__ == '__main__':
  app.run(main)
