# Build deep learning model for learning geodesic perserving descriptor.

load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(default_visibility = [
    "//vr/perception:__subpackages__",
])

pytype_strict_library(
    name = "loss_lib",
    srcs = ["loss_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow",
        "//vr/perception/deepholodeck/human_correspondence/deep_visual_descriptor:utils_lib",
        "//vr/perception/deepholodeck/human_correspondence/optical_flow:utils_lib",
    ],
)

pytype_strict_library(
    name = "geodesic_feature_network",
    srcs = ["geodesic_feature_network.py"],
    srcs_version = "PY3",
    deps = [
        ":loss_lib",
        "//research/vision/piedpiper/brain/python/ops:flow_ops",
        "//third_party/py/tensorflow",
        "//vr/perception/deepholodeck/human_correspondence/deep_visual_descriptor:utils_lib",
        "//vr/perception/deepholodeck/human_correspondence/deep_visual_descriptor/network:resunet_extractor_lib",
    ],
)

py_strict_test(
    name = "loss_lib_test",
    srcs = ["loss_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":loss_lib",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "geodesic_feature_network_test",
    srcs = ["geodesic_feature_network_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":geodesic_feature_network",
        "//third_party/py/tensorflow",
    ],
)
