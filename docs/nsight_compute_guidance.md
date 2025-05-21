# Using Nsight Compute (ncu) for Deep-Dive Kernel Analysis

## 1. Introduction to Nsight Compute

NVIDIA Nsight Compute (`ncu`) is a powerful command-line tool and GUI application designed for interactive kernel profiling on NVIDIA GPUs. It allows developers to collect detailed performance data from CUDA kernels, helping to identify bottlenecks and optimize GPU code.

**When to Use `ncu`:**
Use `ncu` when you need to perform an in-depth analysis of specific CUDA kernels. This is typically after you have identified performance-critical kernels through higher-level profiling tools. `ncu` provides low-level metrics, hardware counter information, and guidance for kernel optimization.

**Nsight Compute (`ncu`) vs. Nsight Systems (`nsys`):**
*   **Nsight Systems (`nsys`):** Provides a system-wide view of performance. It helps understand the interaction between CPU and GPU, identify synchronization issues, and pinpoint which kernels or GPU operations are taking the most time at a high level. It's excellent for understanding the overall application timeline and finding "hot" kernels.
*   **Nsight Compute (`ncu`):** Focuses on individual CUDA kernels. Once `nsys` (or another profiler like `torch.profiler`) has helped you identify a specific kernel that needs optimization, `ncu` is the tool to dissect that kernel's performance in detail. It tells you *why* a particular kernel is slow (e.g., memory latency, instruction stalls, low occupancy).

In short: `nsys` for system-level and application flow analysis; `ncu` for deep-dive analysis of specific kernels.

## 2. Typical Workflow

The typical workflow for using `ncu` involves identifying candidate kernels and then running `ncu` to collect detailed metrics.

### Identifying Candidate Kernels

Before diving into `ncu`, you need to know which kernels are worth analyzing.
*   **Using `torch.profiler` (via `edd_utils.ProfCallback`):**
    When `edd_utils.ProfCallback` is used with its standard PyTorch profiler backend (i.e., `enable_nsight_systems=False`), it generates output files, including `key_averages.txt`. This file lists CUDA kernels sorted by metrics like total duration, which can directly point you to the most time-consuming kernels. Look for kernels with high "self_cuda_time_total" or "cuda_time_total".
*   **Using Nsight Systems (`nsys`):**
    If you've run your application with `nsys profile ...` (potentially using the `enable_nsight_systems=True` mode in `edd_utils.ProfCallback` to generate NVTX ranges), the Nsight Systems GUI or `nsys stats` command can show you a timeline of CUDA kernels and their durations. Identify the longest-running kernels from this report.

### Running Nsight Compute

Once you have a list of candidate kernel names (or a regular expression matching them), you can use `ncu`.

**Basic Command:**
```bash
ncu -o <report_name>.ncu-rep python your_application.py <your_application_args>
```
This command will profile all kernels launched by `your_application.py` and save the report. However, profiling all kernels can be time-consuming and generate very large files. It's almost always better to target specific kernels.

**Essential Command-Line Options:**

*   **`--kernel-name <regex_or_substring>` or `-k <regex_or_substring>`:**
    This is the most important option for focusing `ncu`'s efforts. It restricts profiling to kernels whose names match the provided regular expression or substring.
    *Example:* `ncu -k my_custom_kernel_v2 ...`
    *Example (regex):* `ncu -k "gemm|conv"`
    You can get kernel names from the `key_averages.txt` (from `torch.profiler`) or `nsys` reports.

*   **`--section <section_name>`:**
    `ncu` collects data in "sections," which are predefined sets of metrics. Profiling many sections can be slow.
    *   To list available sections: `ncu --query-sections`
    *   Start with general sections like `SpeedOfLight` or `ComputeWorkloadAnalysis` and then drill down into more specific ones like `MemoryWorkloadAnalysis`, `InstructionStats`, etc., based on initial findings.
    *Example:* `ncu --section SpeedOfLight -k my_kernel ...`

*   **`--set <basic|full>`:**
    *   `basic`: Collects a small, predefined set of metrics. Useful for a quick overview.
    *   `full`: Collects all available metrics for the target kernel(s). This can be very slow and is generally used when you need exhaustive information for a very specific kernel.

*   **`-c, --launch-count <count>`:**
    Profiles only the specified number of launches for each targeted kernel. Useful if a kernel is launched many times but you only need to analyze a few instances.
    *Example:* `ncu -c 3 -k my_kernel ...` (profiles the first 3 launches of `my_kernel`)

*   **`--launch-skip <count>`:**
    Skips the specified number of initial launches for each targeted kernel. Useful for avoiding profiling during warm-up phases.
    *Example:* `ncu --launch-skip 10 -c 1 -k my_kernel ...` (skips the first 10 launches, then profiles the 11th)

*   **`--nvtx` (or `--nvtx-include <regex>`, `--nvtx-exclude <regex>`):**
    If your PyTorch code (or `edd_utils.ProfCallback`) generates NVTX ranges (e.g., `torch.cuda.nvtx.range_push/pop`), `ncu` can correlate its kernel data with these ranges. This provides valuable context, helping you understand which part of your application logic launched the profiled kernel.
    *   `--nvtx`: Enables NVTX collection.
    *   When using `edd_utils.ProfCallback` with `enable_nsight_systems=False`, NVTX ranges are typically generated by PyTorch's profiler. `ncu` can then pick these up.

*   **Targeting MPI Ranks (Advanced):**
    When profiling MPI applications where multiple ranks might launch kernels:
    *   You might need a wrapper script that launches `ncu` only for the specific rank(s) you want to profile.
    *   Some MPI implementations or launch systems might allow prefixing the `ncu` command directly for specific ranks.
    *   `ncu` has options like `--launch-MPI-rank <rank>` but this often requires `ncu` to launch the MPI application itself, which can be complex to set up. A common approach is:
        ```bash
        # Example wrapper script logic (rank_script.sh)
        # if [ "$OMPI_COMM_WORLD_RANK" = "0" ]; then
        #   ncu -o report_rank0 python my_script.py
        # else
        #   python my_script.py
        # fi
        # mpirun -np 4 ./rank_script.sh
        ```
    This is an advanced topic; consult the Nsight Compute documentation for specific MPI setups.

### Analyzing Results

*   **Nsight Compute GUI:**
    The primary way to analyze `.ncu-rep` files is by opening them in the Nsight Compute GUI. This interface provides detailed views of metrics, source code correlation (if available), and actionable advice for optimization.
*   **Command-Line Analysis:**
    *   `ncu --query-metrics <metric_name_regex>`: Queries available metrics.
    *   `ncu --print-summary <report_name>.ncu-rep`: Prints a basic summary of the collected data to the console.
    *   `ncu --export <output_file_name.csv> --page <details_page_name> <report_name>.ncu-rep`: Exports specific data pages to CSV for custom analysis.

## 3. Tips for Effective Profiling with `ncu`

*   **Profile Selectively:** `ncu` has significant overhead. Avoid profiling all kernels or too many kernels at once. Focus on one or a small group of related kernels identified by higher-level profilers.
*   **Iterative Section Analysis:** Start with broad metric sections (e.g., `SpeedOfLight`, `ComputeWorkloadAnalysis`, `MemoryWorkloadAnalysis` available via `ncu --list-sections`). Based on the bottlenecks identified (e.g., if memory bound), drill down with more specific sections (e.g., `L1Cache`, `MemoryWorkloadAnalysis_Chart_L1`).
*   **Use Launch Count/Skip:** For kernels that are called many times, use `--launch-count` and `--launch-skip` to isolate specific invocations, especially if behavior changes over time or during different phases of your application.
*   **Minimize Interference:** `ncu` is designed for deep kernel inspection. Its overhead means it's not suitable for continuous, application-level performance monitoring.

## 4. Interaction with `edd_utils`

`edd_utils` can facilitate the use of `ncu` primarily through its kernel identification capabilities:

*   **Kernel Identification:**
    *   When `ProfCallback` is used with its standard PyTorch profiler backend (`enable_nsight_systems=False`), the generated `key_averages.txt` file is a direct source for identifying top CUDA kernels by duration. These kernel names can be fed directly into `ncu`'s `-k` or `--kernel-name` option.
    *   If you use `ProfCallback` with `enable_nsight_systems=True`, it helps in generating NVTX ranges that make Nsight Systems (`nsys`) reports more readable. You can then identify target kernels from the `nsys` report for subsequent `ncu` analysis.

*   **Minimizing Profiler Interference for `ncu` Runs:**
    When you are running `ncu` to profile a specific kernel, it is generally best to minimize interference from other profiling tools to ensure `ncu` gets the most accurate view of the kernel's intrinsic performance.
    *   **Disable `ProfCallback`:** If you are performing an `ncu` run, consider disabling `ProfCallback` entirely (e.g., by commenting out its setup or using a conditional flag in your training script).
    *   **NVTX Context for `ncu`:** If you *do* want the NVTX context from PyTorch to be available in your `ncu` report (which can be very helpful), you need PyTorch's NVTX generation to be active.
        *   You can achieve this by keeping `ProfCallback` active but ensuring it doesn't produce its own bulky trace files. If `ProfCallback` is configured with `enable_nsight_systems=True` and you are *not* running under `nsys profile` but directly with `ncu`, the `ProfCallback` will issue a warning and disable itself, which is not what you want for NVTX capture by `ncu`.
        *   A better approach for NVTX capture by `ncu` using `ProfCallback` is to ensure `enable_nsight_systems=False`, but configure the `torch.profiler.profile` instance within `ProfCallback` to have `on_trace_ready=None`. This would require a modification to `edd_utils` or manually setting up `torch.profiler.profile` to only generate NVTX ranges.
        *   Alternatively, you can manually add `torch.cuda.nvtx.range_push/pop` around relevant sections of your code.
    *   The key is that `ncu` can capture NVTX ranges generated by PyTorch. If `ProfCallback` is also trying to write full PyTorch profiler traces via its `trace_handler`, it might add unnecessary overhead during an `ncu` run.

**Recommendation for `ncu` and `edd_utils`:**
1.  Use `ProfCallback` (standard mode or `nsys` mode) to identify your target kernels.
2.  For the dedicated `ncu` run on those specific kernels:
    *   Ideally, disable `ProfCallback` to minimize overhead.
    *   If NVTX context is crucial for your `ncu` analysis and your application doesn't have many manual NVTX ranges, you might keep `ProfCallback` enabled but ensure it's configured to *only* generate NVTX (e.g., by setting `on_trace_ready=None` in the `torch.profiler.profile` object it uses, or by setting its export options for other artifacts like memory timeline, stacks, key_averages to `False`). This currently requires a minor modification or specific setup of `ProfCallback` not directly exposed by default arguments if you want to avoid all other artifacts from `trace_handler`. A simpler path if `ProfCallback` is unmodified is to just accept the small overhead of it generating NVTX ranges if you're not concerned about its other trace file outputs during the `ncu` run.

By following these guidelines, you can effectively use Nsight Compute to perform detailed kernel analysis and optimize your PyTorch models, with `edd_utils` helping in the initial stages of identifying which kernels to focus on.
