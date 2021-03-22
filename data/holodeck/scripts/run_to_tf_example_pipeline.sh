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
# run_to_tf_example_pipeline.sh --shards=1 --num_sample=2 --output="/tmp/out.sst"

set -e

source gbash.sh || exit

DEFINE_string borg_cell "tp" "The cell to use"
DEFINE_string output "/cns/is-d/home/feitongtan/human_correspondence/hololight/sstables_512/intra_2000_test_0.40_60.sst" "Output SSTable path"
DEFINE_int shards 5 "Number of shards."
DEFINE_string holodeck_data_base_dir "/bigstore/vr-perception-funshot-export/v5" "Holodeck data base directory."
# DEFINE_string sequence_ids "302_20190801_103352,303_20190801_113758,306_20190801_142931,309_20190802_090918,310_20190802_094720,312_20190802_144449,315_20190802_173240,318_20190805_151329,319_20190805_151753,321_20190805_162135,322_20190806_104833,323_20190806_104942,324_20190806_134352,326_20190806_155358,327_20190806_155446,328_20190806_154724,330_20190807_102729,331_20190807_111629,332_20190807_111733,333_20190807_131841,334_20190807_131940,335_20190807_145421,336_20190807_145534,337_20190808_091953,338_20190808_132749,339_20190808_144919,340_20190808_155923,341_20190808_160145,342_20190809_091903,343_20190809_091955,344_20190809_102348,345_20190809_112129,346_20190809_112308,354_20190812_101625,355_20190812_102028,356_20190812_102148,357_20190812_131319,358_20190812_140629,359_20190812_140927,360_20190812_152520,361_20190813_103142,362_20190813_103351,363_20190813_112246,364_20190813_112405,365_20190813_112534,366_20190813_142202,367_20190813_142044,368_20190813_153128,369_20190813_153313,370_20190813_153502" "Comma separated sequence 50 IDs"
DEFINE_string sequence_ids "320_20190805_161446,307_20190801_153556,325_20190806_134535,304_20190801_121714,308_20190801_153706,313_20190802_154021,329_20190806_155211,314_20190802_162759,317_20190805_104322,347_20190809_112459" "10 sequences"
DEFINE_string cameras "C26A" "Comma separated camera IDs"
DEFINE_int image_height 768 "Image height."
DEFINE_int image_width  512 "Image width."
DEFINE_int num_sample 200 "Number to sample."
DEFINE_string fov_range "15.0" "Half of field of view to sample source camera."
DEFINE_string zoom_min "0.40" "Percentage of radius to sample source camera."
DEFINE_string zoom_max "1.1" "Percentage of radius to sample source camera."
DEFINE_string delta_zoom_min "0.40"  "Percentage of radius to sample target camera."
DEFINE_string delta_zoom_max "1.0"  "Percentage of radius to sample target camera."
DEFINE_string delta_fov_range "65.0" "Half of field of view to sample target camera."
DEFINE_string anchor_range "0.3" "The range to shift the source anchor point."
DEFINE_string delta_anchor_range "0.3" "The range to shift the target anchor point."
DEFINE_string tilt_range "0.05" "The range to tilt the camera."

# Parse command line arguments.
gbash::init_google "$@"

if [[ "${GBASH_ARGC}" -ne "0" ]] ; then
  echo "${GBASH_ARGC} unused arguments. This is probably an error:"
  echo "${GBASH_ARGV}"
  exit -1
fi

rabbit --verifiable build --experimental_deps_ok -c opt vr/perception/deepholodeck/human_correspondence/data/holodeck/to_tf_example_pipeline.par
readonly G3BIN="$(rabbit info blaze-bin -c opt --force_python=PY3)"

"${G3BIN}/vr/perception/deepholodeck/human_correspondence/data/holodeck/to_tf_example_pipeline.par" \
  --flume_borg_accounting_charged_user_name="vr-beaming" \
  --output="${FLAGS_output}" \
  --shards="${FLAGS_shards}" \
  --holodeck_data_base_dir="${FLAGS_holodeck_data_base_dir}" \
  --sequence_ids="${FLAGS_sequence_ids}" \
  --cameras="${FLAGS_cameras}" \
  --image_height="${FLAGS_image_height}" \
  --image_width="${FLAGS_image_width}" \
  --num_sample="${FLAGS_num_sample}" \
  --fov_range="${FLAGS_fov_range}" \
  --zoom_range="${FLAGS_zoom_min}" \
  --zoom_range="${FLAGS_zoom_max}" \
  --delta_zoom_range="${FLAGS_delta_zoom_min}" \
  --delta_zoom_range="${FLAGS_delta_zoom_max}" \
  --delta_fov_range="${FLAGS_delta_fov_range}" \
  --anchor_range="${FLAGS_anchor_range}" \
  --delta_anchor_range="${FLAGS_delta_anchor_range}" \
  --tilt_range="${FLAGS_tilt_range}" \
  --flume_borg_user_name="vr-beaming" \
  --flume_borg_cells="${FLAGS_borg_cell}" \
  --flume_use_batch_scheduler=true \
  --flume_batch_scheduler_strategy=RUN_WITHIN \
  --flume_batch_scheduler_start_deadline_secs=60 \
  --flume_clean_up_tmp_dirs=ALWAYS \
  --flume_exec_mode=IN_PROCESS \
  --alsologtostderr \
  --bigstore_tos=AF1

