# Utility functions for generating SMPL dataset.

load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

py_strict_test(
    name = "utils_test",
    srcs = ["utils_test.py"],
    data = [
        "//vr/perception/deepholodeck/human_correspondence/data/utils/test_data:test_mesh_file",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//file/colossus/public:cns",
        "//net/proto2/python/public:use_pure_python",  # fixdeps: keep go/proto_python_default
        "//testing/pybase",
        "//third_party/py/numpy",
    ],
)

pytype_strict_library(
    name = "utils",
    srcs = ["utils.py"],
    srcs_version = "PY3",
    deps = [
        "//pyglib:gfile",
        "//third_party/py/PIL:pil",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
    ],
)

pytype_strict_binary(
    name = "to_tf_example_pipeline",
    srcs = ["to_tf_example_pipeline.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils",
        "//pipeline/flume/py:runner",
        "//pipeline/flume/py/io:sstableio",
        "//pyglib:gfile",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/absl/logging",
        "//third_party/py/apache_beam",
        "//third_party/py/numpy",
        "//third_party/py/scipy",
        "//third_party/py/tensorflow",
        "//util/task/python:error",
        "//vision/sfm/wrappers/python:camera",
        "//vr/perception/deep_relighting/data_generation/python:io_utils",
        "//vr/perception/deepholodeck/human_correspondence/data/utils:data_utils",
    ],
)
