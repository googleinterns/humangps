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

"""A training framework using distributed strategy."""
import os

from absl import app
from absl import flags
import gin.tf

from google3.vr.perception.deepholodeck.human_correspondence import train_eval_lib_local

flags.DEFINE_enum('mode', None, ['cpu', 'gpu'],
                  'Distributed strategy approach.')
flags.DEFINE_string('base_folder', None, 'Path to checkpoints/summaries.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
flags.DEFINE_multi_string('gin_configs', None, 'Gin config files.')

FLAGS = flags.FLAGS


def main(argv):
  del argv  # Unused.

  gin.parse_config_files_and_bindings(
      config_files=FLAGS.gin_configs,
      bindings=FLAGS.gin_bindings,
      skip_unknown=True)

  base_folder = FLAGS.base_folder
  base_folder = os.path.join(base_folder, 'train')
  train_eval_lib_local.train_pipeline(
      training_mode=FLAGS.mode,
      base_folder=base_folder,
      dataset_params=gin.REQUIRED,
      lr_params=gin.REQUIRED,
      batch_size=gin.REQUIRED,
      n_iterations=gin.REQUIRED)


if __name__ == '__main__':
  flags.mark_flag_as_required('mode')
  flags.mark_flag_as_required('base_folder')
  flags.mark_flag_as_required('gin_configs')
  app.run(main)
