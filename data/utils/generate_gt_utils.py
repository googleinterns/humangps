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

"""Utility functions for generating ground truth."""

from typing import Tuple
import numpy as np

from google3.vision.sfm.wrappers.python import camera as vision_sfm_camera
from google3.vr.perception.stickercap.gt_generation.python import gt_generation

_NEAR_CLIPPING = 1e-6
_FAR_CLIPPING = 1e6


def render_geodesic_map(mesh: Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray],
                        camera: vision_sfm_camera.Camera,
                        num_sample: int = 4) -> Tuple[np.ndarray, np.ndarray]:
  """Samples pixels from a mesh rendering and computes their geodesic maps.

  Randomly sample several foreground pixels from the rendering of the mesh in
  camera view, and computes geodesic distance from all foreground pixels of the
  rendering to each sampled pixels, finally return 2D coordinates of the sampled
  pixels and geodesic distance which is stored in maps.

  Args:
    mesh: A mesh stored with vertices, vertex faces, texture coordinates and
      texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where N1,
      N2, N3, N4 are the number of vertices, vertex faces, texture coordinates
      and texture-coordinate faces respectively.
    camera: The input camera for rendering mesh.
    num_sample: The number of geodesic maps to generate.

  Returns:
    2D coordinates of sampled pixels: a tensor of [N,2], where N is the number
      of sampled vertices.
    Geodesic maps: a tensor of [H,W,N], where H,W are height and width of the
      rendering, and N is the number of sampled pixels.
  """
  _, faces_img, barycentric_img = gt_generation.render_geometry_buffer(
      mesh[:2], camera, _NEAR_CLIPPING, _FAR_CLIPPING)

  height, width = faces_img.shape[:2]
  x_coords, y_coords = np.meshgrid(range(width), range(height), indexing='xy')
  xy_coords = np.stack((x_coords, y_coords), axis=-1)

  # Find all foreground pixels.
  fg_mask = faces_img >= 0
  fg_xy_coords = xy_coords[fg_mask]
  fg_faces = faces_img[fg_mask]
  fg_barycentrics = barycentric_img[fg_mask]

  # Randomly sample pixels.
  fg_pixel_num = np.sum(fg_mask)
  selected_pixel_idx = np.random.choice(
      range(fg_pixel_num), num_sample, replace=False)
  source_xy_coords = fg_xy_coords[selected_pixel_idx]
  source_faces = fg_faces[selected_pixel_idx]
  source_barycentrics = fg_barycentrics[selected_pixel_idx]

  # Set all foreground pixels to target pixels.
  target_face = fg_faces
  target_barycentric = fg_barycentrics

  # Compute geodesic maps for all target pixels to each sampled pixels.
  geodesics = gt_generation.generate_ground_truth_geodesics(
      mesh[:2], source_faces, source_barycentrics, target_face,
      target_barycentric)
  # Set pixels with infinite geodesic distance to zero.
  geodesics[np.isinf(geodesics)] = 0
  geodesic_maps = []
  for idx in range(num_sample):
    # Store geodesic distance in a map.
    geodesic_map = np.zeros([height, width])
    geodesic_map[fg_mask] = geodesics[idx, :]
    geodesic_maps.append(geodesic_map)

  geodesic_maps = np.stack(geodesic_maps, axis=-1)

  return (source_xy_coords, geodesic_maps)


def render_cross_geodesic_map(
    mesh_source: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    mesh_target: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    camera_source: vision_sfm_camera.Camera,
    camera_target: vision_sfm_camera.Camera,
    num_sample: int = 4) -> Tuple[np.ndarray, np.ndarray]:
  """Samples pixels from a mesh rendering and computes cross-view geodesic maps.

  Randomly sample several foreground pixels from the rendering of the source
  mesh in source camera view, and compute geodesic distance from all pixels
  of the target-mesh rendering in target camers view to each sampled source
  pixels, finally return 2D coordinates of the sampled pixels and cross-view
  geodesic distance which is stored in maps. Notice that source mesh and target
  mesh should have the same topology.


  Args:
    mesh_source: A mesh stored with vertices, vertex faces, texture coordinates
      and texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where
      N1, N2, N3, N4 are the number of vertices, vertex faces, texture
      coordinates and texture-coordinate faces respectively.
    mesh_target: A mesh which has the same topology as mesh_source.
    camera_source: The source camera where we source pixels.
    camera_target: The target camera where we compute geodesic maps.
    num_sample: The number of geodesic maps to generate.

  Returns:
    2D coordinates of sampled pixels: a tensor of [N,2], where N is the number
      of sampled pixels.
    Geodesic maps: a tensor of [H,W,N], where H,W are height and width of the
      rendering, and N is the number of sampled pixels.
  """
  _, source_face_map, source_barycentric_map = gt_generation.render_geometry_buffer(
      mesh_source[:2], camera_source, _NEAR_CLIPPING, _FAR_CLIPPING)
  _, target_face_map, target_barycentric_map = gt_generation.render_geometry_buffer(
      mesh_target[:2], camera_target, _NEAR_CLIPPING, _FAR_CLIPPING)

  # Find all foreground pixels in source camera view.
  height, width = source_face_map.shape[:2]
  x_coords, y_coords = np.meshgrid(range(width), range(height), indexing='xy')
  xy_coords = np.stack([x_coords, y_coords], axis=-1)
  source_fg_mask = source_face_map >= 0
  source_fg_xy_coords = xy_coords[source_fg_mask]
  source_fg_faces = source_face_map[source_fg_mask]
  source_fg_barycentrics = source_barycentric_map[source_fg_mask]

  # Randomly sample pixels in source camera view.
  source_fg_pixel_num = np.sum(source_fg_mask)
  selected_pixel_idx = np.random.choice(
      range(source_fg_pixel_num), num_sample, replace=False)
  source_xy_coords = source_fg_xy_coords[selected_pixel_idx]
  source_faces = source_fg_faces[selected_pixel_idx]
  source_barycentrics = source_fg_barycentrics[selected_pixel_idx]

  # Set target pixels in target camera view.
  target_fg_mask = target_face_map >= 0
  target_faces = target_face_map[target_fg_mask]
  target_barycentrics = target_barycentric_map[target_fg_mask]

  # Compute geodesic maps for all target pixels to each sampled pixels.
  geodesics = gt_generation.generate_ground_truth_geodesics(
      mesh_target[:2], source_faces, source_barycentrics, target_faces,
      target_barycentrics)
  # Set pixels with infinite geodesic distance to zero.
  geodesics[np.isinf(geodesics)] = 0
  geodesic_maps = []
  for idx in range(num_sample):
    # Store cross-view geodesic distance in a map.
    geodesic_map = np.zeros([height, width])
    geodesic_map[target_face_map >= 0] = geodesics[idx, :]
    geodesic_maps.append(geodesic_map)

  geodesic_maps = np.stack(geodesic_maps, axis=-1)

  return (source_xy_coords, geodesic_maps)


def render_2d_joints(joints_3d: np.ndarray,
                     camera: vision_sfm_camera.Camera) -> np.ndarray:
  """Projects 3D joints to 2D image by given a camera."""
  joints_2d, _ = camera.Project(joints_3d)
  return joints_2d


def generate_geometric_matrix(
    mesh: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    camera: vision_sfm_camera.Camera,
    num_sample: int = 128) -> Tuple[np.ndarray, np.ndarray]:
  """Samples two sets of pixels and generates a geodesic distance matrix.

  Randomly sample two sets of pixels from a rendering of the mesh in camera
  view, and compute geodesic distance from one set of pixels to the other
  set, finally return 2D coordinates of the sampled pixels and a geodesic matrix
  which stores the geodesic distance from one set of sampled pixels to the other
  set.


  Args:
    mesh: A mesh stored with vertices, vertex faces, texture coordinates and
      texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where N1,
      N2, N3, N4 are the number of vertices, vertex faces, texture coordinates
      and texture-coordinate faces respectively.
    camera: The input camera for rendering mesh.
    num_sample: The number of a set of sampled pixels.

  Returns:
    Coordinates of two set of sampled pixels: A tensor of size [2*N,
      2], where N is the number of a set of sampled pixels.
    The corresponding geodesic matrix: A tensor of size [N, N], where N is the
      number of a set of sampled pixels.
  """
  _, face_map, barycentric_map = gt_generation.render_geometry_buffer(
      mesh[:2], camera, _NEAR_CLIPPING, _FAR_CLIPPING)

  height, width = face_map.shape[:2]

  x_coords, y_coords = np.meshgrid(range(width), range(height), indexing='xy')
  xy_coords = np.stack((x_coords, y_coords), axis=-1)

  fg_mask = face_map >= 0
  fg_xy_coords = xy_coords[fg_mask]
  fg_faces = face_map[fg_mask]
  fg_barycentrics = barycentric_map[fg_mask]

  # Randomly sample num_sample * 2 pixels
  fg_pixel_num = np.sum(fg_mask)
  selected_pixel_idx = np.random.choice(
      range(fg_pixel_num), num_sample * 2, replace=False)

  xy_coords = fg_xy_coords[selected_pixel_idx]
  source_faces = fg_faces[selected_pixel_idx[:num_sample]]
  source_barycentrics = fg_barycentrics[selected_pixel_idx[:num_sample]]
  target_faces = fg_faces[selected_pixel_idx[num_sample:]]
  target_barycentrics = fg_barycentrics[selected_pixel_idx[num_sample:]]

  geodesic_matrix = gt_generation.generate_ground_truth_geodesics(
      mesh[:2], source_faces, source_barycentrics, target_faces,
      target_barycentrics)
  # Set pixels with infinite geodesic distance to zero.
  geodesic_matrix[np.isinf(geodesic_matrix)] = 0

  return (xy_coords, geodesic_matrix)


def render_over_segmentation(
    mesh: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    face_segmentation: np.ndarray,
    camera: vision_sfm_camera.Camera,
) -> np.ndarray:
  """Renders an over-segmentation map for a human mesh.

  Args:
    mesh: A mesh stored with vertices, vertex faces, texture coordinates and
      texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where N1,
      N2, N3, N4 are the number of vertices, vertex faces, texture coordinates
      and texture-coordinate faces respectively.
    face_segmentation: A tensor of size N specifying the segmentation index for
      each face, where N is the number of vertex faces.
    camera: The input camera for rendering mesh.

  Returns:
    A segmentation image of size H x W specifying the segmentation index for
      each foreground pixel, where H x W is the size of segmentation map.
  """
  _, face_map, _ = gt_generation.render_geometry_buffer(mesh[:2], camera,
                                                        _NEAR_CLIPPING,
                                                        _FAR_CLIPPING)
  # Set pixels with minus face index to zero.
  face_map[face_map < 0] = 0
  segmentation_map = face_segmentation[face_map]
  segmentation_map[face_map < 0] = 0
  return segmentation_map
