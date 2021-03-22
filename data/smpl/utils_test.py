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

"""Tests for google3.vr.perception.deepholodeck.human_correspondence.data.smpl.utils."""

import numpy as np

from google3.testing.pybase import googletest
from google3.vr.perception.deepholodeck.human_correspondence.data.smpl import utils

_SMPL_MODEL_PARAMETERS_PATH = '/cns/tp-d/home/yindaz/human_correspondence/dataset/surreal/datagen/pkl/basicModel_f_lbs_10_207_0_v1.0.0_py3.pkl'
_TEST_TEXTURE_MAP_PATH = (
    'vr/perception/deepholodeck/human_correspondence/data/utils/test_data/texture_map.jpg'
)


class UtilsTest(googletest.TestCase):

  def test_smpl_model(self):
    smpl_model_generator = utils.SMPLModel(_SMPL_MODEL_PARAMETERS_PATH)
    pose = (np.random.rand(24, 3) - 0.5)
    beta = (np.random.rand(10,) - 0.5) * 2.5
    smpl_model_generator.set_params(pose=pose, beta=beta)
    smpl_vertex = smpl_model_generator.get_vertex()

    self.assertSequenceEqual(smpl_vertex.shape, (6890, 3))

  def test_smpl_pair_generator(self):
    sample_pair = ({
        'pose_trans': {
            'poses': (np.random.rand(24, 3) - 0.5),
            'trans': np.array((0.0, 0.0, 0.0)),
        },
        'subject': {
            'shape': (np.random.rand(10,) - 0.5) * 2.5,
            'gender': 'female',
            'texture_dir': _TEST_TEXTURE_MAP_PATH
        }
    }, {
        'pose_trans': {
            'poses': (np.random.rand(24, 3) - 0.5),
            'trans': np.array((0.0, 0.0, 0.0)),
        },
        'subject': {
            'shape': (np.random.rand(10,) - 0.5) * 2.5,
            'gender': 'male',
            'texture_dir': _TEST_TEXTURE_MAP_PATH
        }
    })
    smpl_model_generator = utils.SMPLPairGenerator()

    smpl_mesh_pair = smpl_model_generator.generate_smpl_pair(sample_pair)
    source_mesh = smpl_mesh_pair[0]['mesh']
    target_mesh = smpl_mesh_pair[1]['mesh']
    texture_map = smpl_mesh_pair[0]['texture_map']
    joints_3d = smpl_mesh_pair[0]['joints_3d']

    self.assertSequenceEqual(source_mesh[0].shape, (6890, 3))
    self.assertSequenceEqual(source_mesh[1].shape, target_mesh[1].shape)
    self.assertSequenceEqual(source_mesh[2].shape, target_mesh[2].shape)
    self.assertSequenceEqual(source_mesh[3].shape, target_mesh[3].shape)
    self.assertSequenceEqual(texture_map.shape, (8192, 8192, 3))
    self.assertSequenceEqual(joints_3d.shape, (24, 3))


if __name__ == '__main__':
  googletest.main()
