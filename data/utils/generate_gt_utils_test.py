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

"""Tests for google3.vr.perception.deepholodeck.human_correspondence.data.utils.generate_gt_utils."""
import pickle

import numpy as np

from google3.pyglib import gfile
from google3.testing.pybase import googletest
from google3.vision.sfm.wrappers.python import camera as vision_sfm_camera
from google3.vr.perception.deepholodeck.human_correspondence.data.utils import generate_gt_utils

_NUM_SAMPLE = 5
_NUM_JOINTS = 20
_IMAGE_WIDTH = 550
_IMAGE_HEIGHT = 600
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


class GenerateGtUtilsTest(googletest.TestCase):

  def test_render_geodesic_map(self):
    mesh = _load_test_mesh(_MESH_TEST_FILE)
    camera = _get_test_camera()
    source_xy_coords, geodesic_maps = generate_gt_utils.render_geodesic_map(
        mesh, camera, _NUM_SAMPLE)

    self.assertSequenceEqual(source_xy_coords.shape, (_NUM_SAMPLE, 2))
    self.assertSequenceEqual(geodesic_maps.shape,
                             (_IMAGE_HEIGHT, _IMAGE_WIDTH, _NUM_SAMPLE))

  def test_render_cross_geodesic_map(self):
    mesh1 = _load_test_mesh(_MESH_TEST_FILE)
    mesh2 = _load_test_mesh(_MESH_TEST_FILE)
    camera1 = _get_test_camera()
    camera2 = _get_test_camera()

    source_xy_coords, geodesic_maps = generate_gt_utils.render_cross_geodesic_map(
        mesh1, mesh2, camera1, camera2, _NUM_SAMPLE)

    self.assertSequenceEqual(source_xy_coords.shape, (_NUM_SAMPLE, 2))
    self.assertSequenceEqual(geodesic_maps.shape,
                             (_IMAGE_HEIGHT, _IMAGE_WIDTH, _NUM_SAMPLE))

  def test_generate_geometric_matrix(self):
    mesh = _load_test_mesh(_MESH_TEST_FILE)
    camera = _get_test_camera()

    xy_coords, geodesic_matrix = generate_gt_utils.generate_geometric_matrix(
        mesh, camera, _NUM_SAMPLE)

    self.assertSequenceEqual(xy_coords.shape, (_NUM_SAMPLE * 2, 2))
    self.assertSequenceEqual(geodesic_matrix.shape, (_NUM_SAMPLE, _NUM_SAMPLE))

  def test_render_2d_joints(self):
    tolerance = 1e-12
    joints_3d = np.random.rand(_NUM_JOINTS, 3)
    camera = _get_test_camera()
    if joints_3d.shape[-1] == 3:  # Convert 3D points to 4D homogeneous points.
      homogeneous_joints_3d = np.insert(joints_3d, 3, 1.0, axis=-1)
    projection_matrix = camera.ToProjectionMatrix()
    gt_joints_2d = np.matmul(projection_matrix,
                             np.transpose(homogeneous_joints_3d))
    gt_joints_2d = np.transpose(gt_joints_2d)
    gt_joints_2d = gt_joints_2d[:, :2] / gt_joints_2d[:, 2:]

    joints_2d = generate_gt_utils.render_2d_joints(joints_3d, camera)

    self.assertSequenceEqual(joints_2d.shape, (_NUM_JOINTS, 2))
    self.assertAlmostEqual(
        np.linalg.norm(joints_2d - gt_joints_2d), 0.0, delta=tolerance)

  def test_render_over_segmentation(self):
    mesh = _load_test_mesh(_MESH_TEST_FILE)
    face_segmentation = np.arange(mesh[1].shape[0])
    camera = _get_test_camera()

    segmentation_map = generate_gt_utils.render_over_segmentation(
        mesh, face_segmentation, camera)

    self.assertSequenceEqual(segmentation_map.shape,
                             (_IMAGE_HEIGHT, _IMAGE_WIDTH))


if __name__ == '__main__':
  googletest.main()
