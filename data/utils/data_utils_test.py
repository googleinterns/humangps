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

"""Tests for google3.vr.perception.deepholodeck.human_correspondence.data.utilities.render_utils."""

import pickle

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from google3.pyglib import gfile
from google3.vision.sfm.wrappers.python import camera as vision_sfm_camera
from google3.vr.perception.deepholodeck.human_correspondence.data.utils import data_utils

_IMAGE_WIDTH = 24
_IMAGE_HEIGHT = 36
_NUM_MAP_SAMPLE = 4
_NUM_MATRIX_SAMPLE = 12
_MESH_TEST_FILE = (
    'vr/perception/deepholodeck/human_correspondence/data/utils/test_data/test_mesh.pickle'
)


def _get_test_camera(rotation=None):
  """Returns a vision sfm camera wrapper for testing."""
  camera = vision_sfm_camera.Camera()
  if rotation is None:
    camera.SetOrientationFromRotationMatrix(
        np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]]))
  elif rotation == 'clockwise':
    camera.SetOrientationFromRotationMatrix(
        np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1]]))
  elif rotation == 'counter_clockwise':
    camera.SetOrientationFromRotationMatrix(
        np.array([[0, 1, 0], [-1, 0, 0], [0, 0, 1]]))
  camera.SetImageSize(_IMAGE_WIDTH, _IMAGE_HEIGHT)
  camera.SetPrincipalPoint(_IMAGE_WIDTH / 2, _IMAGE_HEIGHT / 2)
  camera.SetFocalLength(300)
  camera.SetPosition(np.array([0, 0, -5]))
  return camera


def _load_test_mesh(path):
  """Load a mesh obj for testing.

  Args:
    path: A path to a test file.

  Returns:
    A test mesh stored with vertices, vertex faces, texture coordinates
      and texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where
      N1, N2, N3, N4 are the number of vertices, vertex faces, texture
      coordinates and texture-coordinate faces respectively.
  """

  with gfile.Open(path, 'rb') as f:
    mesh = pickle.load(f)
  return mesh


class RenderUtilsTest(tf.test.TestCase, parameterized.TestCase):

  def test_add_texture_coordinate_face_to_mesh(self):
    tolerance = 1e-12
    vertex = np.zeros((10, 3), np.float32)
    vertex_face = np.zeros((10, 3), np.int32)
    uv_coordinate = np.zeros((10, 2), np.float32)
    mesh = (vertex, vertex_face, uv_coordinate)

    extended_mesh = data_utils.add_texture_coordinate_face_to_mesh(mesh)

    self.assertSequenceEqual(extended_mesh[0].shape, mesh[0].shape)
    self.assertSequenceEqual(extended_mesh[1].shape, mesh[1].shape)
    self.assertSequenceEqual(extended_mesh[2].shape, mesh[2].shape)
    self.assertSequenceEqual(extended_mesh[3].shape, mesh[1].shape)
    self.assertAlmostEqual(
        np.linalg.norm(extended_mesh[0] - mesh[0]), 0.0, delta=tolerance)
    self.assertAlmostEqual(
        np.linalg.norm(extended_mesh[1] - mesh[1]), 0.0, delta=tolerance)
    self.assertAlmostEqual(
        np.linalg.norm(extended_mesh[2] - mesh[2]), 0.0, delta=tolerance)
    self.assertAlmostEqual(
        np.linalg.norm(extended_mesh[3] - mesh[1]), 0.0, delta=tolerance)

  def test_render_rgb_msaa(self):
    mesh = _load_test_mesh(_MESH_TEST_FILE)
    uv_texture = np.zeros((20, 20, 3), np.uint8)
    camera = _get_test_camera()
    msaa_factor = 8.0

    rgb_rendering = data_utils.render_rgb_msaa(mesh, uv_texture, camera,
                                               msaa_factor)
    self.assertEqual(rgb_rendering.shape, (_IMAGE_HEIGHT, _IMAGE_WIDTH, 3))

  def test_add_rgb_rendering(self):
    feature = {}
    mesh = _load_test_mesh(_MESH_TEST_FILE)
    uv_texture = np.zeros((20, 20, 3), np.uint8)
    camera = _get_test_camera()
    msaa_factor = 8.0
    camera_tag = 'source'

    data_utils.add_rgb_rendering(feature, mesh, uv_texture, camera, camera_tag,
                                 msaa_factor)

    self.assertIn('image/uint8_png_bytes/source/rgb', list(feature.keys()))

  def test_add_rgb_rendering_holodeck(self):
    feature = {}
    mesh = _load_test_mesh(_MESH_TEST_FILE)
    uv_texture = np.zeros((10, 6, 3), np.float32)
    warp_field = np.zeros((_IMAGE_HEIGHT, _IMAGE_WIDTH, 2), np.float32)
    camera = _get_test_camera()
    camera_tag = 'source'

    data_utils.add_rgb_rendering_holodeck(feature, uv_texture, warp_field, mesh,
                                          camera, camera_tag)

    self.assertIn('image/uint8_png_bytes/source/rgb', list(feature.keys()))

  def test_add_foregroung_mask(self):
    feature = {}
    mesh = _load_test_mesh(_MESH_TEST_FILE)
    camera = _get_test_camera()
    camera_tag = 'source'

    data_utils.add_foregroung_mask(feature, mesh, camera, camera_tag)

    self.assertIn('image/uint8_png_bytes/source/mask', list(feature.keys()))

  def test_add_optical_flow_rendering(self):
    feature = {}
    source_mesh = _load_test_mesh(_MESH_TEST_FILE)
    target_mesh = _load_test_mesh(_MESH_TEST_FILE)
    source_camera = _get_test_camera()
    target_camera = _get_test_camera()
    camera_tag = 'source'

    flow, flow_mask = data_utils.add_optical_flow_rendering(
        feature, source_mesh, target_mesh, source_camera, target_camera,
        camera_tag)

    self.assertIn('image/uint16_png_bytes/source/forward_flow',
                  list(feature.keys()))
    self.assertIn('image/uint8_png_bytes/source/flow_mask',
                  list(feature.keys()))
    self.assertEqual(flow.shape, (_IMAGE_HEIGHT, _IMAGE_WIDTH, 2))
    self.assertEqual(flow_mask.shape, (_IMAGE_HEIGHT, _IMAGE_WIDTH))

  def test_add_geodesic_maps_rendering(self):
    feature = {}
    mesh = _load_test_mesh(_MESH_TEST_FILE)
    camera = _get_test_camera()
    camera_tag = 'source'

    data_utils.add_geodesic_maps_rendering(feature, mesh, camera, camera_tag,
                                           _NUM_MAP_SAMPLE)

    self.assertIn('vector/float/source/geo_centers', list(feature.keys()))
    self.assertIn('image/uint16_png_bytes/source/geodesic_maps',
                  list(feature.keys()))

  def test_add_cross_geodesic_maps_rendering(self):
    feature = {}
    source_mesh = _load_test_mesh(_MESH_TEST_FILE)
    target_mesh = _load_test_mesh(_MESH_TEST_FILE)
    source_camera = _get_test_camera()
    target_camera = _get_test_camera()
    camera_tag = 'source'

    data_utils.add_cross_geodesic_maps_rendering(feature, source_mesh,
                                                 target_mesh, source_camera,
                                                 target_camera, camera_tag)

    self.assertIn('vector/float/source/cross_geo_centers', list(feature.keys()))
    self.assertIn('image/uint16_png_bytes/source/cross_geodesic_maps',
                  list(feature.keys()))

  def test_add_geodesic_matrix_rendering(self):
    feature = {}
    mesh = _load_test_mesh(_MESH_TEST_FILE)
    camera = _get_test_camera()
    camera_tag = 'source'

    data_utils.add_geodesic_matrix_rendering(feature, mesh, camera, camera_tag,
                                             _NUM_MATRIX_SAMPLE)

    self.assertIn('vector/float/source/geo_keypoints', list(feature.keys()))
    self.assertIn('image/uint16_png_bytes/source/geodesic_matrix',
                  list(feature.keys()))

  def test_add_segmentation_rendering(self):
    feature = {}
    mesh = _load_test_mesh(_MESH_TEST_FILE)
    face_segmentation = np.arange(mesh[1].shape[0])
    camera = _get_test_camera()
    camera_tag = 'source'

    data_utils.add_segmentation_rendering(feature, mesh, face_segmentation,
                                          camera, camera_tag)

    self.assertIn('image/uint16_png_bytes/source/seg_map', list(feature.keys()))


if __name__ == '__main__':
  tf.test.main()
