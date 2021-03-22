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

"""Utility function for generating renderpeople dataset.

RenderPeople is a library of scanned photorealistic 3D people, which is
higher-fidelity than SMPL model. Here is the link: https://renderpeople.com/
"""
from typing import Dict, Any, Tuple

import numpy as np
from PIL import Image
from google3.pyglib import gfile


def load_obj_to_mesh(
    path: str,
    zero_translation: bool = True
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
  """Load a mesh in obj format by giving a path.

  Args:
    path: A path to a mesh obj file.
    zero_translation: A bool flag. If it is true, the position of the mesh will
      be set as zero-centered.

  Returns:
    A mesh stored with vertices, vertex faces, texture coordinates
      and texture-coordinate faces of size [N1,3], [N2,3], [N3,2], [N4,3], where
      N1, N2, N3, N4 are the number of vertices, vertex faces, texture
      coordinates and texture-coordinate faces respectively.
  """
  with gfile.Open(path, 'r') as f:
    obj_text = f.read()

  lines = obj_text.splitlines()
  vs = []
  vts = []
  faces = []

  for line in lines:
    elem = line.split()
    if elem:
      if elem[0] == 'v':
        v_ = [float(elem[1]), float(elem[2]), float(elem[3])]
        vs.append(v_)

      if elem[0] == 'vt':
        vt_ = [float(elem[1]), float(elem[2])]
        vts.append(vt_)

      if elem[0] == 'f':
        face_ = []
        for i in range(1, len(elem)):
          face_v_, face_vt_, _ = elem[i].split('/')
          face_v_vt = [int(face_v_), int(face_vt_)]
          face_.append(face_v_vt)
        faces.append(face_)

  vs = np.asarray(vs)
  if zero_translation:
    vs[:, 0] = vs[:, 0] - np.mean(vs[:, 0])
    vs[:, 1] = vs[:, 1] - np.mean(vs[:, 1])
    vs[:, 2] = vs[:, 2] - np.mean(vs[:, 2])

  vts = np.asarray(vts)
  faces = np.asarray(faces) - 1
  faces_v = faces[:, :, 0].astype(np.int32)
  faces_vt = faces[:, :, 1].astype(np.int32)

  mesh = (vs, faces_v, vts, faces_vt)

  return mesh


def load_renderpeople_mesh(obj_path: str, tex_path: str) -> Dict[str, Any]:
  """load a renderpeople mesh with texture map."""
  renderpeople_mesh = {}
  mesh = load_obj_to_mesh(obj_path)
  tex_img = np.array(Image.open(gfile.GFile(tex_path)))
  renderpeople_mesh = {
      'mesh': mesh,
      'texture_map': tex_img,
  }
  return renderpeople_mesh
