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

"""Tests for google3.vr.perception.deepholodeck.human_correspondence.data.renderpeople.utils."""

from google3.vr.perception.deepholodeck.human_correspondence.data.renderpeople import utils
from google3.testing.pybase import googletest

_TEST_OBJ_PATH = (
    'vr/perception/deepholodeck/human_correspondence/data/utils/test_data/renderpeople.obj'
)

_TEST_TEXTURE_MAP_PATH = (
    'vr/perception/deepholodeck/human_correspondence/data/utils/test_data/texture_map.jpg'
)


class UtilsTest(googletest.TestCase):

  def test_load_obj_to_mesh(self):
    test_mesh = utils.load_obj_to_mesh(_TEST_OBJ_PATH)

    self.assertSequenceEqual(test_mesh[0].shape, (4, 3))
    self.assertSequenceEqual(test_mesh[1].shape, (4, 3))
    self.assertSequenceEqual(test_mesh[2].shape, (4, 2))
    self.assertSequenceEqual(test_mesh[3].shape, (4, 3))

  def test_load_renderpeople_mesh(self):
    renderpeople_mesh = utils.load_renderpeople_mesh(_TEST_OBJ_PATH,
                                                     _TEST_TEXTURE_MAP_PATH)
    test_mesh = renderpeople_mesh['mesh']
    test_texture_map = renderpeople_mesh['texture_map']

    self.assertSequenceEqual(test_mesh[0].shape, (4, 3))
    self.assertSequenceEqual(test_mesh[1].shape, (4, 3))
    self.assertSequenceEqual(test_mesh[2].shape, (4, 2))
    self.assertSequenceEqual(test_mesh[3].shape, (4, 3))
    self.assertSequenceEqual(test_texture_map.shape, (8192, 8192, 3))


if __name__ == '__main__':
  googletest.main()
