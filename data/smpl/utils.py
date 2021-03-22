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

"""Generate SMPL model by take pose, shape, transilation parameters."""

import pickle
from typing import Any, Dict, Tuple

import numpy as np
from PIL import Image

from google3.pyglib import gfile

_BASICMODEL_M_DIR = '/cns/tp-d/home/yindaz/human_correspondence/dataset/surreal/datagen/pkl/basicModel_m_lbs_10_207_0_v1.0.0_py3.pkl'
_BASICMODEL_F_DIR = '/cns/tp-d/home/yindaz/human_correspondence/dataset/surreal/datagen/pkl/basicModel_f_lbs_10_207_0_v1.0.0_py3.pkl'
_UV_FACE_DIR = '/cns/tp-d/home/yindaz/human_correspondence/dataset/surreal/datagen/smpl_data/uv_face_append.txt'
_TMP_DIR = '/cns/tp-d/home/yindaz/human_correspondence/dataset/surreal/tmp/'


class SMPLModel:
  """SMPL model generator which generates SMPL mesh from SMPL parameters."""

  def __init__(self, model_path: str):
    """Initialize SMPL model generator.

    Args:
      model_path: Path to load SMPL model hyperparameters.
    """
    with gfile.Open(model_path, 'rb') as f:
      bytes_object = f.read()
      params = pickle.loads(bytes_object)

    self._joint_regressor = params['J_regressor']
    self._weights = params['weights']
    self._posedirs = params['posedirs']
    self._v_template = params['v_template']
    self._shapedirs = params['shapedirs']
    self._faces = params['f']
    self._kintree_table = params['kintree_table']

    id_to_col = {
        self._kintree_table[1, i]: i
        for i in range(self._kintree_table.shape[1])
    }
    self.parent = {
        i: id_to_col[self._kintree_table[0, i]]
        for i in range(1, self._kintree_table.shape[1])
    }

    self.pose_shape = [24, 3]
    self.beta_shape = [10]
    self.trans_shape = [3]

    self.pose = np.zeros(self.pose_shape)
    self.beta = np.zeros(self.beta_shape)
    self.trans = np.zeros(self.trans_shape)

    self.verts = None
    self.joint = None
    self.rotation = None

    self.update()

  def set_params(self,
                 pose: np.ndarray = None,
                 beta: np.ndarray = None,
                 translation: np.ndarray = np.array([0.0, 0.0, 0.0])):
    """Set pose, shape, and/or translation parameters of SMPL model.

    Args:
      pose: Also known as 'theta', a [24,3] matrix indicating child joint
        rotation relative to parent joint. For root joint it's global
        orientation. Represented in a axis-angle format.
      beta: Parameter for model shape. A vector of shape [10]. Coefficients for
        PCA component. Only 10 components were released by MPI.
      translation: Global translation of shape [3].

    Returns:
      Updated vertices.
    """
    if pose is not None:
      self.pose = pose
    if beta is not None:
      self.beta = beta
    if translation is not None:
      self.trans = translation
    self.update()
    return self.verts

  def update(self):
    """Called automatically when parameters are updated."""
    # Adjust shape according to beta.
    v_shaped = self._shapedirs.dot(self.beta) + self._v_template
    # Joints location.
    self.joint = self._joint_regressor.dot(v_shaped)
    pose_cube = self.pose.reshape((-1, 1, 3))
    # Rotation matrix for each joint.
    self.rotation = self.rodrigues(pose_cube)
    i_cube = np.broadcast_to(
        np.expand_dims(np.eye(3), axis=0), (self.rotation.shape[0] - 1, 3, 3))
    lrotmin = (self.rotation[1:] - i_cube).ravel()
    # Adjust shape in zero pose according to pose parameters.
    v_posed = v_shaped + self._posedirs.dot(lrotmin)
    # Apply world transformation for each joint.
    g_matrix = np.empty((self._kintree_table.shape[1], 4, 4))
    g_matrix[0] = self.with_zeros(
        np.hstack((self.rotation[0], self.joint[0, :].reshape([3, 1]))))
    for i in range(1, self._kintree_table.shape[1]):
      g_matrix[i] = g_matrix[self.parent[i]].dot(
          self.with_zeros(
              np.hstack([
                  self.rotation[i],
                  ((self.joint[i, :] - self.joint[self.parent[i], :]).reshape(
                      [3, 1]))
              ])))

    g_matrix = g_matrix - self.pack(
        np.matmul(
            g_matrix,
            np.hstack([self.joint, np.zeros([24, 1])]).reshape([24, 4, 1])))
    # Transformation of each vertex
    t_matrix = np.tensordot(self._weights, g_matrix, axes=[[1], [0]])
    rest_shape_h = np.hstack((v_posed, np.ones([v_posed.shape[0], 1])))
    vertex = np.matmul(t_matrix,
                       rest_shape_h.reshape([-1, 4, 1])).reshape([-1, 4])[:, :3]
    self.verts = vertex + self.trans.reshape([1, 3])
    self.jtr = np.stack([g[:3, 3] for g in g_matrix],
                        axis=0) + self.trans.reshape([1, 3])

  def rodrigues(self, axis_angle: np.ndarray):
    """Rodrigues formula that turns axis-angle vector into rotation matrix.

    Args:
      axis_angle: Axis-angle rotation vector of shape [batch_size, 1, 3].

    Returns:
      Rotation matrix of shape [batch_size, 3, 3].
    """
    theta = np.linalg.norm(axis_angle, axis=(1, 2), keepdims=True)
    # Avoid zero divide
    theta = np.maximum(theta, np.finfo(np.float64).tiny)
    r_hat = axis_angle / theta
    cos = np.cos(theta)
    z_stick = np.zeros(theta.shape[0])
    m = np.dstack([
        z_stick, -r_hat[:, 0, 2], r_hat[:, 0, 1], r_hat[:, 0, 2], z_stick,
        -r_hat[:, 0, 0], -r_hat[:, 0, 1], r_hat[:, 0, 0], z_stick
    ]).reshape([-1, 3, 3])
    i_cube = np.broadcast_to(
        np.expand_dims(np.eye(3), axis=0), [theta.shape[0], 3, 3])
    r_hat_transpose = np.transpose(r_hat, axes=[0, 2, 1])
    dot = np.matmul(r_hat_transpose, r_hat)
    rotation_matrix = cos * i_cube + (1 - cos) * dot + np.sin(theta) * m
    return rotation_matrix

  def with_zeros(self, matrix34: np.ndarray):
    """Append a [0, 0, 0, 1] vector to a [3, 4] matrix.

    Args:
      matrix34: Matrix to be appended.

    Returns:
      Matrix after appending of shape [4,4]
    """
    return np.vstack((matrix34, np.array([[0.0, 0.0, 0.0, 1.0]])))

  def pack(self, matrix41: np.ndarray):
    """Append zero matrices of shape to vectors of shape in a batched manner.

    Args:
      matrix41: Matrices to be appended of shape [batch_size, 4, 1]

    Returns:
      Matrix of shape [batch_size, 4, 4] after appending.
    """
    return np.dstack((np.zeros((matrix41.shape[0], 4, 3)), matrix41))

  def get_vertex(self):
    """Return vertices of SMPL Model."""
    return self.verts.copy()

  def get_3d_joints(self):
    """Return 3D joints of SMPL Model."""
    return self.jtr.copy()

  def save_to_obj(self, path: str, material_path: str = None):
    """Save the SMPL model into .obj file."""
    with gfile.Open(path, 'w') as fp:
      if material_path is not None:
        fp.write('mtllib material.mtl\n')
      for vertex in self.verts:
        fp.write('v %f %f %f\n' % (vertex[0], vertex[1], vertex[2]))

      if material_path is not None:
        with gfile.Open(material_path, 'r') as uv_f:
          fp.write(uv_f.read())
      else:
        for face in self._faces + 1:
          fp.write('f %d %d %d\n' % (face[0], face[1], face[2]))


class SMPLPairGenerator:
  """Generate SMPL model pair by giving smpl parameters of source and target."""

  def __init__(self):
    """Initialize SMPL Pair generator."""
    self._basic_model_f_dir = _BASICMODEL_F_DIR
    self._basic_model_m_dir = _BASICMODEL_M_DIR
    self._uv_face_append_dir = _UV_FACE_DIR
    self._tmp_path = _TMP_DIR
    print('SMPL Model Loading ...')
    self._female_model = SMPLModel(self._basic_model_f_dir)
    self._male_model = SMPLModel(self._basic_model_m_dir)
    print('Finished SMPL Model Loadding')
    self.load_uv_text_to_np(self._uv_face_append_dir)

  def load_uv_text_to_np(self, txt_dir: str):
    """Read vertex faces and texture-coordinate faces of SMPL model."""
    with gfile.Open(txt_dir, 'r') as uv_f:
      uv_f_text = uv_f.read()

    lines = uv_f_text.splitlines()
    vts = []
    faces = []

    for line in lines:
      elem = line.split()
      if elem:
        if elem[0] == 'vt':
          vt_ = [float(elem[1]), float(elem[2])]
          vts.append(vt_)
        if elem[0] == 'f':
          face = []
          for i in range(1, len(elem)):
            v_, vt_ = elem[i].split('/')
            v_vt = [int(v_), int(vt_)]
            face.append(v_vt)
          faces.append(face)

    self.vts = np.asarray(vts)
    faces_temp = np.asarray(faces) - 1

    self.faces_v = faces_temp[:, :, 0].astype(np.int32)
    self.faces_vt = faces_temp[:, :, 1].astype(np.int32)

  def generate_smpl(self,
                    frame: Dict[str, Dict[str, Any]],
                    zero_translation=True,
                    dump_obj=False):
    """Generate SMPL model for one frame.

    Args:
      frame: A dict stored the SMPL pose parameters, translation, shape
        parameters and gender.
      zero_translation: A bool flag. If it is true, the position of the mesh
        will be set as zero-centered.
      dump_obj: A bool flag to decide whether to dump mesh obj for test.

    Returns:
      A tensor of size 6890 x 3, stored the vertices of a SMPL model.
      A tensor of size 24 x 3, stored the joints of a SMPL model.
    """
    poses = frame['pose_trans']['poses']
    trans = frame['pose_trans']['trans']

    shape = frame['subject']['shape']
    gender = frame['subject']['gender']

    if gender == 'female':
      smpl_model = self._female_model
    else:
      smpl_model = self._male_model

    if zero_translation:
      smpl_model.set_params(beta=shape, pose=poses)
    else:
      smpl_model.set_params(beta=shape, pose=poses, translation=trans)

    vs = smpl_model.get_vertex()
    joints = smpl_model.get_3d_joints()
    self.vs = vs

    if dump_obj:
      seq_name = frame['pose_info']['seq_name']
      frame_idx = frame['pose_info']['frame_idx']
      output = '%s/%s_%d.obj' % (self._tmp_path, seq_name, frame_idx)
      self.save_to_obj(output)

    return vs, joints

  def generate_smpl_pair(self,
                         sample_pair: Tuple[Dict[str, Any], Dict[str, Any]],
                         zero_translation: bool = True,
                         dump_obj: bool = False):
    """Generate SMPL model for one frame.

    Args:
      sample_pair: A list of two dict stored the SMPL pose parameters,
        translation, shape parameters and gender.
      zero_translation: A bool flag. If it is true, the position of the mesh
        will be set as zero-centered.
      dump_obj: A bool flag to decide whether to dump mesh obj for test.

    Returns:
      A tensor of size 6890 x 3, stored the vertices of a SMPL model.
      A tensor of size 24 x 3, stored the joints of a SMPL model.
    """
    source_frame = sample_pair[0]
    target_frame = sample_pair[1]

    source_vert, source_joints = self.generate_smpl(source_frame,
                                                    zero_translation, dump_obj)
    target_vert, target_joints = self.generate_smpl(target_frame,
                                                    zero_translation, dump_obj)

    source_image = np.array(
        Image.open(gfile.GFile(source_frame['subject']['texture_dir'])))
    target_image = np.array(
        Image.open(gfile.GFile(target_frame['subject']['texture_dir'])))

    # The face index starts from 0.
    source_mesh = {
        'mesh': (source_vert, self.faces_v.copy(), self.vts.copy(),
                 self.faces_vt.copy()),
        'joints_3d': source_joints,
        'texture_map': source_image,
        'pelvis_rotation': source_frame['pose_trans']['poses'][:3],
    }

    target_mesh = {
        'mesh': (target_vert, self.faces_v.copy(), self.vts.copy(),
                 self.faces_vt.copy()),
        'joints_3d': target_joints,
        'texture_map': target_image,
        'pelvis_rotation': target_frame['pose_trans']['poses'][:3],
    }

    return (source_mesh, target_mesh)

  def save_to_obj(self, path: str):
    """Save the SMPL model into .obj file.

    Args:
      path: Path to save.
    """
    with gfile.Open(path, 'w') as fp:
      fp.write('mtllib material.mtl\n')
      for vertex in self.vs:
        fp.write('v %f %f %f\n' % (vertex[0], vertex[1], vertex[2]))

      for texture_coord in self.vts:
        fp.write('vt %f %f\n' % (texture_coord[0], texture_coord[1]))
      fp.write('usemtl lambert1\n')
      fp.write('s 1\n')

      for vertex_face, texture_face in zip(self.faces_v + 1, self.faces_vt + 1):
        fp.write('f %d/%d %d/%d %d/%d\n' %
                 (vertex_face[0], texture_face[0], vertex_face[1],
                  texture_face[1], vertex_face[2], texture_face[2]))
