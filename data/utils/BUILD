# Utility functions for generating human correspondence dataset.
load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(default_visibility = [
    "//vr/perception:__subpackages__",
])

pytype_strict_library(
    name = "generate_gt_utils",
    srcs = ["generate_gt_utils.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "//vision/sfm/wrappers/python:camera",
        "//vr/perception/stickercap/gt_generation/python:gt_generation",
    ],
)

pytype_strict_library(
    name = "data_utils",
    srcs = ["data_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":generate_gt_utils",
        "//third_party/py/cvx2",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_graphics/image/color_space:srgb",
        "//vision/sfm/wrappers/python:camera",
        "//vr/perception/deepholodeck/relightable_textures/data:data_utils",
        "//vr/perception/stickercap/gt_generation/python:gt_generation",
    ],
)

py_strict_test(
    name = "generate_gt_utils_test",
    srcs = ["generate_gt_utils_test.py"],
    data = [
        "//vr/perception/deepholodeck/human_correspondence/data/utils/test_data:test_mesh_file",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":generate_gt_utils",
        "//net/proto2/python/public:use_pure_python",  # fixdeps: keep go/proto_python_default
        "//pyglib:gfile",
        "//testing/pybase",
        "//third_party/py/numpy",
        "//vision/sfm/wrappers/python:camera",
    ],
)

py_strict_test(
    name = "data_utils_test",
    srcs = ["data_utils_test.py"],
    data = [
        "//vr/perception/deepholodeck/human_correspondence/data/utils/test_data:test_mesh_file",
    ],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":data_utils",
        "//pyglib:gfile",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//vision/sfm/wrappers/python:camera",
    ],
)
