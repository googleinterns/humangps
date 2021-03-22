#!/bin/bash
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


source gbash.sh || exit

DEFINE_string mode "gpu" "Training mode."
DEFINE_string base_folder "/tmp/test01/" "Folder to save log and checkpoint."
DEFINE_string model_gin "./gin_config/model_config.gin" "Gin configuration for model."
DEFINE_string data_gin "./gin_config/data_config.gin" "Gin configuration for data."
DEFINE_string train_eval_gin "./gin_config/train_eval_config_local.gin" "Gin configuration for train_eval."
# Exit if anything fails.
set -e


fileutil mkdir -p "$FLAGS_base_folder"
fileutil mkdir -p "$FLAGS_base_folder/gin_config"

fileutil cp -f "${FLAGS_model_gin}" "$FLAGS_base_folder/gin_config"
fileutil cp -f "${FLAGS_data_gin}" "$FLAGS_base_folder/gin_config"
fileutil cp -f "${FLAGS_train_eval_gin}" "$FLAGS_base_folder/gin_config"

bazel run -c opt --copt=-mavx --define cuda_target_sm75=1 --config=cuda \
  train_main_local -- \
  --base_folder="$FLAGS_base_folder" \
  --mode="$FLAGS_mode" \
  --gin_configs="${FLAGS_model_gin}" \
  --gin_configs="${FLAGS_data_gin}" \
  --gin_configs="${FLAGS_train_eval_gin}" \
  --gfs_user=vr-beaming \
  --alsologtostderr
