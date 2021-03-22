# Build deep learning models for dense human correspondence.

load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_binary", "pytype_strict_library")

pytype_strict_library(
    name = "dataset_lib",
    srcs = ["dataset_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/tensorflow:tensorflow_google",
        "//vr/perception/deepholodeck/human_correspondence/optical_flow:utils_lib",
        "//vr/perception/tensorflow/data:dataset_loading",
    ],
)

pytype_strict_library(
    name = "train_eval_lib_local",
    srcs = ["train_eval_lib_local.py"],
    srcs_version = "PY3",
    deps = [
        ":dataset_lib",
        "//third_party/py/absl/logging",
        "//third_party/py/gin/tf",
        "//third_party/py/tensorflow",
        "//vr/perception/deepholodeck/human_correspondence/deep_visual_descriptor/model:geodesic_feature_network",
        "//vr/perception/deepholodeck/human_correspondence/optical_flow/pwcnet:pwcnet_lib",
        "//vr/perception/deepholodeck/human_correspondence/optical_flow/raft:raft_lib",
    ],
)

py_strict_test(
    name = "dataset_lib_test",
    srcs = ["dataset_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset_lib",
        "//third_party/py/tensorflow",
    ],
)

py_strict_test(
    name = "train_eval_lib_test",
    timeout = "long",
    srcs = ["train_eval_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":dataset_lib",
        ":train_eval_lib",
        "//third_party/py/absl/testing:flagsaver",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
        "//vr/perception/deepholodeck/human_correspondence/deep_visual_descriptor/model:geodesic_feature_network",
        "//vr/perception/deepholodeck/human_correspondence/optical_flow/pwcnet:pwcnet_lib",
        "//vr/perception/deepholodeck/human_correspondence/optical_flow/raft:raft_lib",
    ],
)

pytype_strict_binary(
    name = "train_main_local",
    srcs = ["train_main_local.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":train_eval_lib_local",
        "//pyglib:file_util",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/gin/tf",
    ],
)

pytype_strict_binary(
    name = "eval_main_local",
    srcs = ["eval_main_local.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":train_eval_lib_local",
        "//pyglib:file_util",
        "//third_party/py/absl:app",
        "//third_party/py/absl/flags",
        "//third_party/py/gin/tf",
    ],
)
