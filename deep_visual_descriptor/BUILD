# Build deep descriptor leanrning framework.
load("//devtools/python/blaze:strict.bzl", "py_strict_test")
load("//devtools/python/blaze:pytype.bzl", "pytype_strict_library")

package(default_visibility = [
    "//vr/perception:__subpackages__",
])

pytype_strict_library(
    name = "utils_lib",
    srcs = ["utils_lib.py"],
    srcs_version = "PY3",
    deps = [
        "//research/vision/piedpiper/brain/python/ops:flow_ops",
        "//third_party/py/tensorflow",
        "//vr/perception/deepholodeck/human_correspondence/optical_flow:utils_lib",
    ],
)

py_strict_test(
    name = "utils_lib_test",
    srcs = ["utils_lib_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":utils_lib",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
    ],
)
