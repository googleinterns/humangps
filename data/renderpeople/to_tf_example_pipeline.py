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
"""A pipeline to create SSTables of tf.Examples from human correspondence data.

See run_to_tf_example_pipeline.sh for example usage.
"""
import os
import pickle
import random
import sys
from typing import Any, Dict, Iterable, Text, Tuple

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
from google3.vision.sfm.wrappers.python import camera as vision_sfm_camera
from google3.vr.perception.deep_relighting.data_generation.python import io_utils
from google3.vr.perception.deepholodeck.human_correspondence.data.renderpeople import utils as renderpeople_utils
from google3.vr.perception.deepholodeck.human_correspondence.data.utils import data_utils as human_data_utils

flags.DEFINE_string('output', '/tmp/output.sst', 'Output SSTable path.')
flags.DEFINE_integer('shards', 10000, 'Number of shards.')
flags.DEFINE_string('sample_pkl_path', None,
                    'The path of pickle for sample list.')
flags.DEFINE_string('camera_pkl_path', None,
                    'The path for pickle for camera list.')
flags.DEFINE_integer('num_example', 100, 'Data to generate.')
flags.DEFINE_integer('num_point_sample', 100, 'Number of points to sample.')

FLAGS = flags.FLAGS

_MAX_GEODESIC = 5.0
_RANDOM_ORDER = True


class ProtoPathToTFExampleFn(beam.DoFn):
  """Beam DoFn that generates a tf.train.Example."""

  def __init__(self, image_height: int, image_width: int,
               num_point_sample: int):
    """Initializes the DoFn.

    Args:
      image_height: The height of the images to be added.
      image_width: The width of the images to be added.
      num_point_sample: Number of points to sample per example.
    """
    self._image_height = image_height
    self._image_width = image_width
    self._num_point_sample = num_point_sample

  def process(
      self, sample_and_camera: Tuple[Tuple[Dict[str, Any], Dict[str, Any]],
                                     Tuple[Any, Any], int]
  ) -> Iterable[Tuple[Text, tf.train.Example]]:
    """Computes and yields a (key, tf.Example) tuple from a path and camera.

    Args:
      sample_and_camera: A tuple containing SMPL parameters, and the camera
        parameters is to be generated.

    Yields:
      A tuple of the SSTable key name and the tf.Example.
    """
    io_utils.set_swiftshader_library_path()
    sample_pair, camera_pair, idx = sample_and_camera
    mesh1 = renderpeople_utils.load_renderpeople_mesh(
        sample_pair[0]['obj_path'], sample_pair[0]['tex_path'])
    mesh2 = renderpeople_utils.load_renderpeople_mesh(
        sample_pair[1]['obj_path'], sample_pair[1]['tex_path'])

    (source_camera_pb, target_camera_pb) = camera_pair
    source_camera = vision_sfm_camera.Camera.FromProto(source_camera_pb)
    target_camera = vision_sfm_camera.Camera.FromProto(target_camera_pb)

    seq_name1 = sample_pair[0]['obj_path'][-10:]
    seq_name2 = sample_pair[1]['obj_path'][-10:]
    frame_idx1 = sample_pair[0]['tex_path'][-5:]
    frame_idx2 = sample_pair[1]['tex_path'][-5:]

    try:
      feature = {}
      # Add RGB, Image are in [0, 255].
      human_data_utils.add_rgb_rendering(
          feature,
          mesh1['mesh'],
          mesh1['texture_map'],
          source_camera,
          'source',
          msaa_factor=8.0)
      human_data_utils.add_rgb_rendering(
          feature,
          mesh2['mesh'],
          mesh2['texture_map'],
          target_camera,
          'target',
          msaa_factor=8.0)

      # Add mask
      source_mask = human_data_utils.add_foregroung_mask(
          feature, mesh1['mesh'], source_camera, 'source')
      _ = human_data_utils.add_foregroung_mask(feature, mesh2['mesh'],
                                               target_camera, 'target')

      # Add dense ground truth.
      _, flow_mask = human_data_utils.add_optical_flow_rendering(
          feature, mesh1['mesh'], mesh2['mesh'], source_camera, target_camera,
          'source')
      if np.sum(flow_mask) / np.sum(source_mask) < 0.4:
        return

      # Add geodesic maps
      human_data_utils.add_geodesic_maps_rendering(feature, mesh1['mesh'],
                                                   source_camera, 'source')
      human_data_utils.add_geodesic_maps_rendering(feature, mesh2['mesh'],
                                                   target_camera, 'target')

      # add cross geodesic maps
      human_data_utils.add_cross_geodesic_maps_rendering(
          feature, mesh1['mesh'], mesh2['mesh'], source_camera, target_camera,
          'source')
      human_data_utils.add_cross_geodesic_maps_rendering(
          feature, mesh2['mesh'], mesh1['mesh'], target_camera, source_camera,
          'target')

      # add geodesic matrix
      human_data_utils.add_geodesic_matrix_rendering(feature, mesh1['mesh'],
                                                     source_camera, 'source')
      human_data_utils.add_geodesic_matrix_rendering(feature, mesh2['mesh'],
                                                     target_camera, 'target')

      # Computes TF.Example.
      example = tf.train.Example(features=tf.train.Features(feature=feature))
      if _RANDOM_ORDER:
        key = str(
            random.randint(0, int(1e9))
        ) + ',' + seq_name1 + ',' + frame_idx1 + ',' + seq_name2 + ',' + frame_idx2
        yield (key, example)
      else:
        key = '%11d' % idx
        yield (key, example)

    except (ValueError, IOError, RuntimeError, error.StatusNotOk) as e:
      # These logging messages will show up in the logs of the worker tasks.
      logging.warning('Element %s,%s,%s,%s skipped due to exception %s',
                      seq_name1, frame_idx1, seq_name2, frame_idx2, str(e))
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
  with gfile.Open(FLAGS.sample_pkl_path, 'rb') as f:
    bytes_object = f.read()
    sample_list = pickle.loads(bytes_object)

  logging.info('%d samples loaded in total.', len(sample_list))

  with gfile.Open(FLAGS.camera_pkl_path, 'rb') as f:
    bytes_object = f.read()
    camera_list = pickle.loads(bytes_object)

  logging.info('%d cameras loaded in total.', len(camera_list))

  random.shuffle(sample_list)
  random.shuffle(camera_list)

  sample_and_camera = []
  for i in range(FLAGS.num_example):
    sample_and_camera.append((sample_list[i], camera_list[i], i))

  logging.info('%d data to process in total.', len(sample_and_camera))

  sample_and_camera_pcol = root | 'create_input' >> beam.Create(
      sample_and_camera)

  key_and_tf_example = (
      sample_and_camera_pcol
      | 'convert_to_tf_example' >> beam.ParDo(
          ProtoPathToTFExampleFn(
              image_height=768,
              image_width=1024,
              num_point_sample=FLAGS.num_point_sample,
          )))

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
