load(
    "//tensorflow/tools/test:performance.bzl",
    "cuda_py_benchmark_test",
)

package(
    # copybara:uncomment default_applicable_licenses = ["//tensorflow:license"],
    default_visibility = ["//tensorflow:internal"],
    licenses = ["notice"],
)

cuda_py_benchmark_test(
    name = "kpi_benchmark_test",
    size = "medium",
    srcs = ["kpi_benchmark_test.py"],
    tags = [
        "no_windows",  #  b/141617449
        "optonly",
    ],
    deps = [
        "//tensorflow:tensorflow_py_no_contrib",
        "//tensorflow/python/eager:benchmarks_test_base",
        "//tensorflow/python/eager:context",
        "//tensorflow/python/profiler:trace",
    ],
)
