# Implementation of deep feature extractors.

load("//devtools/python/blaze:strict.bzl", "py_strict_test")

package(default_visibility = [
    "//vr/perception:__subpackages__",
])

py_library(
    name = "resunet_extractor_lib",
    srcs = ["resunet_extractor_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//third_party/py/numpy",
        "//third_party/py/tensorflow",
        "//third_party/py/tensorflow_addons",
    ],
)

py_strict_test(
    name = "resunet_extractor_lib_test",
    timeout = "long",
    srcs = ["resunet_extractor_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":resunet_extractor_lib",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
    ],
)
