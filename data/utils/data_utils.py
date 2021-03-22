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

"""Utility functions for data generation."""

from typing import Dict, Text, Tuple

from cvx2 import latest as cv2
import numpy as np
import tensorflow as tf
from tensorflow_graphics.image.color_space import srgb

from google3.vision.sfm.wrappers.python import camera as vision_sfm_camera
from google3.vr.perception.deepholodeck.human_correspondence.data.utils import generate_gt_utils
from google3.vr.perception.deepholodeck.relightable_textures.data import data_utils
from google3.vr.perception.stickercap.gt_generation.python import gt_generation

_NEAR_CLIPPING = 1e-6
_FAR_CLIPPING = 1e6
_MAX_GEODESIC = 5.0
_UINT_16_FACTOR = 65535
_UINT_8_FACTOR = 255


def add_texture_coordinate_face_to_mesh(
    mesh: Tuple[np.ndarray, np.ndarray, np.ndarray]
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Add texture-coordinate faces to a mesh for rendering ground truth.

  Some meshes may only store vertex faces since their vertex faces and
  texture-coordinate faces share the same value. In order to make them
  maintaining the same format for rendering, we add texture-coordinate faces for
  them.

  Args:
    mesh: A mesh stored with vertices, vertex faces and texture coordinates of
      size [N1,3], [N2,3], [N3,2], where N1, N2, N3 are the number of vertices,
      vertex faces and texture coordinates respectively.

  Returns:
    A mesh for rendering ground truth, which stores vertices, vertex faces,
      texture coordinates and texture-coordinate faces of size [N1,3], [N2,3],
      [N3,2], [N4,3], where N1, N2, N3, N4 are the number of vertices, vertex
      faces, texture coordinates and texture-coordinate faces respectively.
  """
  faces_vt = mesh[1].copy()
  extended_mesh = (*mesh, faces_vt)
  return extended_mesh


def render_rgb_msaa(mesh: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
                    texture_map: np.ndarray, camera: vision_sfm_camera.Camera,
                    msaa_factor: float) -> np.ndarray:
  """Renders color image using multisample anti-aliasing.

  Renders a color image using multisample anti-aliasing. More details can be
  found in https://en.wikipedia.org/wiki/Multisample_anti-aliasing.

  Args:
    mesh: A mesh stored with vertices, vertex faces, texture coordinates and
      texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where N1,
      N2, N3, N4 are the number of vertices, vertex faces, texture coordinates
      and texture-coordinate faces respectively.
    texture_map: A uint8 tensor stored texture image of size H x W x 3, where H
      x W is the size of texture image.
    camera: The input camera for rendering mesh.
    msaa_factor: A scale factor for multisampling

  Returns:
    A color image: A tensor of size H x W x 3, where H x W is the size of
      rendered image.
  """
  height, width = camera.ImageShape()
  focal = camera.FocalLength()
  msaa_principal_point_x = camera.PrincipalPointX() * msaa_factor
  msaa_principal_point_y = camera.PrincipalPointY() * msaa_factor
  msaa_height, msaa_width = int(height * msaa_factor), int(width * msaa_factor)
  msaa_focal = focal * msaa_factor
  msaa_camera = camera.Copy()
  msaa_camera.SetImageSize(msaa_width, msaa_height)
  msaa_camera.SetFocalLength(msaa_focal)
  msaa_camera.SetPrincipalPoint(msaa_principal_point_x, msaa_principal_point_y)
  msaa_rgb_rendering = gt_generation.render_from_texture(
      mesh, texture_map, msaa_camera)
  rgb_rendering = cv2.resize(
      msaa_rgb_rendering, (width, height), interpolation=cv2.INTER_AREA)
  return rgb_rendering


def add_rgb_rendering(feature: Dict[Text, tf.train.Feature],
                      mesh: Tuple[np.ndarray, np.ndarray, np.ndarray,
                                  np.ndarray],
                      texture_map: np.ndarray,
                      camera: vision_sfm_camera.Camera,
                      camera_tag: str,
                      msaa_factor: float = 1.0):
  """Add rgb rendering to the feature.

  Args:
    feature: The dict to save tf.train.Feature.
    mesh: A mesh stored with vertices, vertex faces, texture coordinates and
      texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where N1,
      N2, N3, N4 are the number of vertices, vertex faces, texture coordinates
      and texture-coordinate faces respectively.
    texture_map: A tensor stored texture image of size H x W x 3, where H x W is
      the size of texture image.
    camera: The input camera for rendering mesh.
    camera_tag: The name of camera, can be 'source' or 'target'.
    msaa_factor: A supersampling factor for multisampling
  """
  if msaa_factor > 1.0:
    rgb_rendering = render_rgb_msaa(mesh, texture_map, camera, msaa_factor)
  else:
    rgb_rendering = gt_generation.render_from_texture(mesh, texture_map, camera)
  _, face_map, _ = gt_generation.render_geometry_buffer(mesh[:2], camera,
                                                        _NEAR_CLIPPING,
                                                        _FAR_CLIPPING)
  foreground_mask = face_map >= 0
  rgb_rendering[~foreground_mask] = 0
  feature['image/uint8_png_bytes/%s/rgb' % camera_tag] = (
      data_utils.get_image_tf_feature(
          rgb_rendering, dtype=tf.uint8, multiply_factor=1))


def add_rgb_rendering_holodeck(feature: Dict[Text, tf.train.Feature],
                               uv_texture: np.ndarray, warp_field: np.ndarray,
                               mesh: Tuple[np.ndarray, np.ndarray, np.ndarray,
                                           np.ndarray],
                               camera: vision_sfm_camera.Camera,
                               camera_tag: str):
  """Add rgb rendering to the feature for holodeck data.

  Args:
    feature: The dict to save tf.train.Feature.
    uv_texture: The texture map.
    warp_field: The warping field from texture space to image space.
    mesh: A mesh stored with vertices, vertex faces, texture coordinates and
      texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where N1,
      N2, N3, N4 are the number of vertices, vertex faces, texture coordinates
      and texture-coordinate faces respectively.
    camera: The input camera for rendering mesh.
    camera_tag: The name of camera, can be 'source' or 'target'.
  """
  rgb_rendering = data_utils.apply_warping_field(uv_texture, warp_field)
  _, face_map, _ = gt_generation.render_geometry_buffer(mesh[:2], camera,
                                                        _NEAR_CLIPPING,
                                                        _FAR_CLIPPING)
  foreground_map = face_map >= 0
  rgb_rendering[~foreground_map] = 0
  rgb_rendering = srgb.from_linear_rgb(rgb_rendering).numpy()
  feature['image/uint8_png_bytes/%s/rgb' % camera_tag] = (
      data_utils.get_image_tf_feature(
          rgb_rendering, dtype=tf.uint8, multiply_factor=_UINT_8_FACTOR))


def add_foregroung_mask(feature: Dict[Text, tf.train.Feature],
                        mesh: Tuple[np.ndarray, np.ndarray, np.ndarray,
                                    np.ndarray],
                        camera: vision_sfm_camera.Camera, camera_tag: str):
  """Add foreground mask to the feature.

  Args:
    feature: The dict to save tf.train.Feature.
    mesh: A mesh stored with vertices, vertex faces, texture coordinates and
      texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where N1,
      N2, N3, N4 are the number of vertices, vertex faces, texture coordinates
      and texture-coordinate faces respectively.
    camera: The input camera for rendering mesh.
    camera_tag: The name of camera, can be 'source' or 'target'.

  Returns:
    A boolean mask map of size H x W, where H x W is the size of camera image in
      pixels.
  """
  _, face_map, _ = gt_generation.render_geometry_buffer(mesh[:2], camera,
                                                        _NEAR_CLIPPING,
                                                        _FAR_CLIPPING)
  foreground_mask = face_map >= 0
  feature['image/uint8_png_bytes/%s/mask' % camera_tag] = (
      data_utils.get_image_tf_feature(
          foreground_mask, dtype=tf.uint8, multiply_factor=255))
  return foreground_mask


def add_optical_flow_rendering(
    feature: Dict[Text,
                  tf.train.Feature], source_mesh: Tuple[np.ndarray, np.ndarray,
                                                        np.ndarray, np.ndarray],
    target_mesh: Tuple[np.ndarray, np.ndarray, np.ndarray,
                       np.ndarray], source_camera: vision_sfm_camera.Camera,
    target_camera: vision_sfm_camera.Camera,
    camera_tag: str) -> Tuple[np.ndarray, np.ndarray]:
  """Add optical flow rendering to the feature.

  Args:
    feature: The dict to save tf.train.Feature.
    source_mesh: A mesh stored with vertices, vertex faces, texture coordinates
      and texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where
      N1, N2, N3, N4 are the number of vertices, vertex faces, texture
      coordinates and texture-coordinate faces respectively.
    target_mesh: The mesh with the same topology as the source mesh.
    source_camera: The source camera for rendering optical flow.
    target_camera: The target camera for rendering optical flow.
    camera_tag: The name of camera, can be 'source' or 'target'.

  Returns:
    A optical flow map of size H x W x 2, where H x W is camera image in pixels.
    A boolean mask map to indicate co-visible pixels for both view of size H x
      W, where H x W is camera image in pixels.
  """
  flow, flow_mask = gt_generation.generate_ground_truth_flow(
      source_mesh[:2], target_mesh[:2], source_camera, target_camera)

  flow_uint16 = flow * 64 + 2**15

  feature['image/uint16_png_bytes/%s/forward_flow' %
          camera_tag] = data_utils.get_image_tf_feature(
              flow_uint16, dtype=tf.uint16, multiply_factor=1)

  feature['image/uint8_png_bytes/%s/flow_mask' %
          camera_tag] = data_utils.get_image_tf_feature(
              flow_mask, dtype=tf.uint8, multiply_factor=1)
  # flow_mask is stored as 0 or 255 to indicate visibility, we use 127 as a
  # threshold to convert it to a boolean mask map.
  flow_mask = flow_mask > 127
  return flow, flow_mask


def add_geodesic_maps_rendering(
    feature: Dict[Text, tf.train.Feature],
    mesh: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    camera: vision_sfm_camera.Camera,
    camera_tag: str,
    num_sample: int = 4,
):
  """Add geodesic maps rendering to the feature.

  Args:
    feature: The dict to save tf.train.Feature.
    mesh: A mesh stored with vertices, vertex faces, texture coordinates and
      texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where N1,
      N2, N3, N4 are the number of vertices, vertex faces, texture coordinates
      and texture-coordinate faces respectively.
    camera: The camera to render geodesic maps.
    camera_tag: The name of camera, can be 'source' or 'target'.
    num_sample: The number of geodesic maps to generate.
  """
  source_xy_coords, geodesic_maps = generate_gt_utils.render_geodesic_map(
      mesh, camera, num_sample)

  height, width, num_maps = geodesic_maps.shape
  geodesic_map_uint16 = geodesic_maps.reshape([height, width * num_maps
                                              ]) / _MAX_GEODESIC

  feature['vector/float/%s/geo_centers' % camera_tag] = tf.train.Feature(
      float_list=tf.train.FloatList(value=source_xy_coords.flatten().tolist()))

  feature['image/uint16_png_bytes/%s/geodesic_maps' %
          camera_tag] = data_utils.get_image_tf_feature(
              geodesic_map_uint16, dtype=tf.uint16, multiply_factor=65535)


def add_cross_geodesic_maps_rendering(
    feature: Dict[Text, tf.train.Feature],
    source_mesh: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    target_mesh: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    source_camera: vision_sfm_camera.Camera,
    target_camera: vision_sfm_camera.Camera,
    camera_tag: str,
    num_sample: int = 4,
):
  """Add cross-view geodesic map rendering to the feature.

  Args:
    feature: The dict to save tf.train.Feature.
    source_mesh: A mesh stored with vertices, vertex faces, texture coordinates
      and texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where
      N1, N2, N3, N4 are the number of vertices, vertex faces, texture
      coordinates and texture-coordinate faces respectively.
    target_mesh: A mesh which has the same topology as mesh_source.
    source_camera: The source camera where we source pixels.
    target_camera: The target camera where we compute geodesic maps.
    camera_tag: The name of camera, can be 'source' or 'target'.
    num_sample: The number of cross-view geodesic maps to generate.
  """
  xy_coords, cross_geodesic_map = generate_gt_utils.render_cross_geodesic_map(
      source_mesh, target_mesh, source_camera, target_camera, num_sample)

  height, width, num_maps = cross_geodesic_map.shape
  cross_geodesic_map_uint16 = cross_geodesic_map.reshape(
      [height, width * num_maps]) / _MAX_GEODESIC

  feature['vector/float/%s/cross_geo_centers' % camera_tag] = tf.train.Feature(
      float_list=tf.train.FloatList(value=xy_coords.flatten().tolist()))

  feature['image/uint16_png_bytes/%s/cross_geodesic_maps' %
          camera_tag] = data_utils.get_image_tf_feature(
              cross_geodesic_map_uint16,
              dtype=tf.uint16,
              multiply_factor=_UINT_16_FACTOR)


def add_geodesic_matrix_rendering(
    feature: Dict[Text, tf.train.Feature],
    mesh: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    camera: vision_sfm_camera.Camera,
    camera_tag: str,
    num_sample: int = 128,
):
  """Add geodesic matrix rendering to the feature.

  Args:
    feature: The dict to save tf.train.Feature.
    mesh: A mesh stored with vertices, vertex faces, texture coordinates and
      texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where N1,
      N2, N3, N4 are the number of vertices, vertex faces, texture coordinates
      and texture-coordinate faces respectively.
    camera: The input camera for rendering mesh.
    camera_tag: The name of camera, can be 'source' or 'target'.
    num_sample: The number of a set of sampled pixels.
  """
  xy_coords, geodesic_matrix = generate_gt_utils.generate_geometric_matrix(
      mesh, camera, num_sample)

  geodesic_matrix_uint16 = geodesic_matrix / _MAX_GEODESIC

  feature['vector/float/%s/geo_keypoints' % camera_tag] = tf.train.Feature(
      float_list=tf.train.FloatList(value=xy_coords.flatten().tolist()))

  feature['image/uint16_png_bytes/%s/geodesic_matrix' %
          camera_tag] = data_utils.get_image_tf_feature(
              geodesic_matrix_uint16,
              dtype=tf.uint16,
              multiply_factor=_UINT_16_FACTOR)


def add_segmentation_rendering(
    feature: Dict[Text, tf.train.Feature],
    mesh: Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
    face_segmentation: np.ndarray,
    camera: vision_sfm_camera.Camera,
    camera_tag: str,
):
  """Add over segmentation rendering to the feature.

  Args:
    feature: The dict to save tf.train.Feature.
    mesh: A mesh stored with vertices, vertex faces, texture coordinates and
      texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where N1,
      N2, N3, N4 are the number of vertices, vertex faces, texture coordinates
      and texture-coordinate faces respectively.
    face_segmentation: A tensor of size N specifying the segmentation index for
      each face, where N is the number of vertex faces.
    camera: The source camera to render over segmentation.
    camera_tag: The name of camera, can be 'source' or 'target'.
  """
  segmentation_map = generate_gt_utils.render_over_segmentation(
      mesh, face_segmentation, camera)

  feature['image/uint16_png_bytes/%s/seg_map' %
          camera_tag] = data_utils.get_image_tf_feature(
              segmentation_map, dtype=tf.uint16, multiply_factor=1)
