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


# Runs a beam on flume pipeline to convert holodeck data to tf.Examples.
#
# Example:
# run_to_tf_example_pipeline_simple.sh --shards=1 --output="/tmp/out.sst"

set -e

source gbash.sh || exit

DEFINE_string borg_cell "tp" "The cell to use"
DEFINE_string output "/cns/tp-d/home/yindaz/human_correspondence/dataset/renderpeople/sstables_512/test.sst" "Output SSTable path"
DEFINE_int shards 1 "Number of shards."
DEFINE_string sample_pkl_path "/cns/tp-d/home/yindaz/human_correspondence/dataset/renderpeople/pkl/renderpeople_training_900000.pkl" "Samples pkl path."
DEFINE_string camera_pkl_path "/cns/tp-d/home/yindaz/human_correspondence/dataset/camera_param_pkl/camera_pair_900000_512_15.0.pkl" "Camera pkl path."
DEFINE_int num_example 5 "Data to generate."
DEFINE_int num_point_sample 5 "Number of points to sample."

# Parse command line arguments.
gbash::init_google "$@"

if [[ "${GBASH_ARGC}" -ne "0" ]] ; then
  echo "${GBASH_ARGC} unused arguments. This is probably an error:"
  echo "${GBASH_ARGV}"
  exit -1
fi

rabbit --verifiable build -c opt vr/perception/deepholodeck/human_correspondence/data/renderpeople/to_tf_example_pipeline.par

readonly G3BIN="$(rabbit info blaze-bin -c opt --force_python=PY3)"

"${G3BIN}/vr/perception/deepholodeck/human_correspondence/data/renderpeople/to_tf_example_pipeline.par" \
  --flume_borg_accounting_charged_user_name="vr-beaming" \
  --output="${FLAGS_output}" \
  --shards="${FLAGS_shards}" \
  --sample_pkl_path="${FLAGS_sample_pkl_path}" \
  --camera_pkl_path="${FLAGS_camera_pkl_path}" \
  --num_example="${FLAGS_num_example}" \
  --num_point_sample="${FLAGS_num_point_sample}" \
  --flume_borg_user_name="vr-beaming" \
  --flume_borg_cells="${FLAGS_borg_cell}" \
  --flume_use_batch_scheduler=true \
  --flume_batch_scheduler_strategy=RUN_WITHIN \
  --flume_batch_scheduler_start_deadline_secs=60 \
  --flume_clean_up_tmp_dirs=ALWAYS \
  --alsologtostderr \
  --bigstore_anonymous \
  # --flume_exec_mode=IN_PROCESS
