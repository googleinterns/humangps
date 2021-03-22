# Train / eval data generation code from holodeck data.

load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary")

package(default_visibility = [
    "//vr/perception:__subpackages__",
])

pytype_strict_binary(
    name = "to_tf_example_pipeline",
    srcs = ["to_tf_example_pipeline.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        "//cloud/bigstore/util:bigstore_file_register",
        "//file/colossus/public:cns",
        "//file/placer",
        "//pipeline/flume/py:runner",
        "//pipeline/flume/py/io:sstableio",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/apache_beam",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//util/task/python:error",
        "//vr/perception/deep_relighting/data_generation/python:io_utils",
        "//vr/perception/deepholodeck/hololight/data:hololight_data_utils",
        "//vr/perception/deepholodeck/human_correspondence/data/utils:data_utils",
        "//vr/perception/deepholodeck/relightable_textures/data:data_utils",
        "//vr/perception/stickercap/python:holodeck_output_proto_api",
    ],
)
