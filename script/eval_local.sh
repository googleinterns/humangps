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

MODE="cpu" # Training mode.
BASE_FOLDER="./experiment/1/" # Folder to save log and checkpoint.
JOB_NAME="eval_optical_flow_intra"
MODEL_GIN="./gin_config/model_config.gin" # Gin configuration for model.
DATA_GIN="./gin_config/data_config.gin" # Gin configuration for data.
TRAIN_EVAL_GIN="./gin_config/train_eval_config_local.gin" # Gin configuration for train_eval.

# Exit if anything fails.
set -e

echo "python eval_main_local.py --base_folder "$BASE_FOLDER" --mode "$MODE" --job_name "$JOB_NAME" --gin_configs "${MODEL_GIN}" --gin_configs "${DATA_GIN}" --gin_configs "${TRAIN_EVAL_GIN}""
python eval_main_local.py --base_folder "$BASE_FOLDER" --mode "$MODE" --job_name "$JOB_NAME" --gin_configs "${MODEL_GIN}" --gin_configs "${DATA_GIN}" --gin_configs "${TRAIN_EVAL_GIN}"

