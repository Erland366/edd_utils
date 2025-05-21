# Profiler code was taken from https://github.com/pytorch/torchtune/blob/890deab3029eef65f94cedb37fda14479f65f129/torchtune/training/_profiler.py
# But now it is implemented for TrainerCallback
from __future__ import annotations

import linecache
import os
import re
import inspect
import numpy as np
from typing import Optional, Union

__all__ = [
    "trace_handler",
    "get_world_size_and_rank",
    "tensor_to_latex",
    "create_dynamic_function",
    "ProfCallback"
]



def import_libraries(*args):
    for arg in args:
        exec(f"{arg}", globals())

def get_world_size_and_rank():
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        return torch.distributed.get_world_size(), torch.distributed.get_rank()
    else:
        return 1, 0

def trace_handler(
    prof,
    output_dir: str,
    metric: str = "self_cuda_time_total",
    row_limit: int = 25,
    export_memory_timeline: bool = True,
    export_stacks: bool = True,
    export_key_averages: bool = True,
):
    """
    Handles export of artifacts from ``torch.profiler.profile``.

    The following artifacts can be exported:
    - Chrome / TensorBoard trace: Always exported. Viewable via TensorBoard or Perfetto.dev / chrome://tracing.
    - Trace event table: Optionally exported if `export_key_averages` is True.
    - Memory timeline and snapshot.pickle: Optionally exported if `export_memory_timeline` is True and `profile_memory` was enabled in the profiler.
    - Stacks: Optionally exported if `export_stacks` is True and `with_stack` was enabled in the profiler. Viewable as a flamegraph.

    Notes:
    - Each profiling cycle is exported as a sub-directory in `output_dir` (e.g., `iteration_5`, `iteration_10`).
    - In a distributed setting, each artifact will be prefixed with the rank.
    - Memory timeline is only exported for rank 0.

    See PyTorch profiler documentation (https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile) for more details.

    Args:
        prof (torch.profiler.profile): Instance of the torch profiler to use.
        output_dir (str): Directory to store artifacts.
        metric (str): Metric to order the trace event table by. See ``torch.profiler.profile.key_averages().table``.
        row_limit (int): Number of rows to display in the trace event table.
        export_memory_timeline (bool): Whether to export the memory timeline. Defaults to True.
        export_stacks (bool): Whether to export stack traces. Defaults to True.
        export_key_averages (bool): Whether to export the event averages table. Defaults to True.
    """
    import_libraries(
        "import time",
        "import datetime",
        "from torch.profiler import tensorboard_trace_handler",
        "import torch",
    )

    world_size, rank = get_world_size_and_rank()
    curr_trace_dir_name = "iteration_" + str(prof.step_num)
    curr_trace_dir = os.path.join(output_dir, curr_trace_dir_name)
    if not os.path.exists(curr_trace_dir):
        os.makedirs(curr_trace_dir, exist_ok=True)

    # Export chrome / tensorboard trace
    if rank == 0:
        print(f"Dumping traces at step {prof.step_num}")
        # log.info(f"Dumping traces at step {prof.step_num}")
    begin = time.monotonic()

    # Use tensorboard trace handler rather than directly exporting chrome traces since
    # tensorboard doesn't seem to be able to parse traces with prof.export_chrome_trace

    now = datetime.datetime.now()

    exporter = tensorboard_trace_handler(
        curr_trace_dir,
        worker_name=f"r0-{now.year}-{now.month}-{now.day}-{now.hour}-{now.minute}",
        use_gzip=True,
    )
    exporter(prof)

    if rank == 0:
        # log.info(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")
        print(f"Finished dumping traces in {time.monotonic() - begin:.2f} seconds")

    # Memory timeline sometimes fails to export
    if export_memory_timeline and prof.profile_memory:
        if rank == 0:
            try:
                prof.export_memory_timeline(
                    f"{curr_trace_dir}/rank{rank}_memory-timeline.html"
                )
            except Exception as e:
                # log.warn(f" Failed to export memory timeline: {e}")
                print(f"Failed to export memory timeline: {e}")

            torch.cuda.memory._dump_snapshot(
                f"{curr_trace_dir}/rank{rank}_memory_snapshot.pickle"
            )

    # Dump stack traces
    if export_stacks and prof.with_stack:
        prof.export_stacks(f"{curr_trace_dir}/rank{rank}_stacks.txt", metric=metric)

    # Export event averages
    if export_key_averages:
        key_avgs = prof.key_averages(
            group_by_input_shape=prof.record_shapes, group_by_stack_n=5
        ).table(sort_by=metric, row_limit=row_limit)
        with open(f"{curr_trace_dir}/rank{rank}_key_averages.txt", "w") as f:
            print(key_avgs, file=f)

    if rank == 0:
        # log.info(f"Saving profiling results to {curr_trace_dir}")
        print(f"Saving profiling results to {curr_trace_dir}")

    # TODO: Is this necessary?
    # see https://github.com/pytorch/torchtitan/blob/3050098dcee4901d88c712f9e8e9703d1735a29b/torchtitan/profiling.py#L48
    if world_size > 1:
        torch.distributed.barrier()

DEFAULT_TRACE_OPTS: dict = {
    "profile_memory": True,
    "with_stack": True,
    "record_shapes": True,
    "with_flops": True,
}

def tensor_array_to_latex(tensor: Union["torch.Tensor", "jax.Array", np.ndarray]):
    """
    Converts a PyTorch Tensor, JAX Array, or NumPy array of any dimension
    into a LaTeX string representation suitable for inline math or a single row matrix.

    Args:
        tensor: The input tensor or array-like object.

    Returns:
        str: A LaTeX string representing the array elements separated by ' & ' and
             ending with ' \\\\\n'.  Suitable for embedding in LaTeX math environments.
             Numbers are rounded to 3 decimal places.
    """

    if isinstance(tensor, np.ndarray):
        M_np = tensor
    else:
        try:
            import_libraries("import torch")
            if isinstance(tensor, torch.Tensor):
                M_np = tensor.detach().cpu().numpy() 
            else:
                raise TypeError("Input tensor must be a NumPy array, PyTorch Tensor, or JAX Array")
        except ImportError:
            try:
                import_libraries("import jax.numpy as jnp") 
                if isinstance(tensor, jnp.ndarray): 
                    M_np = np.array(tensor) 
                else:
                    raise TypeError("Input tensor must be a NumPy array, PyTorch Tensor, or JAX Array")
            except ImportError:
                raise TypeError("Input tensor must be a NumPy array, PyTorch Tensor, or JAX Array")


    M_np = np.array(M_np).flatten() 
    M_np = np.round(M_np, 3) 
    M_str_list = [f"{x:.3f}" for x in M_np] 
    latex_row = " & ".join(M_str_list) 
    return latex_row + r" \\" + "\n" 


def tensor_matrix_to_latex(tensor: Union["torch.Tensor", "jax.Array", np.ndarray]):
    """
    Converts a PyTorch Tensor, JAX Array, or NumPy array of any dimension
    into a LaTeX string representation of a matrix environment. For dimensions higher than 2,
    it will represent them as stacked matrices.

    Args:
        tensor: The input tensor or array-like object.

    Returns:
        str: A LaTeX string representing the array within a 'bmatrix' environment.
             Rows are separated by ' \\\\\n' and elements in each row by ' & '.
             Numbers are rounded to 3 decimal places.
    """
    if isinstance(tensor, np.ndarray):
        M_np = tensor
    else:
        try:
            import_libraries("import torch")
            if isinstance(tensor, torch.Tensor):
                M_np = tensor.detach().cpu().numpy() # Move to CPU and convert to NumPy
            else:
                raise TypeError("Input tensor must be a NumPy array, PyTorch Tensor, or JAX Array")
        except ImportError:
            try:
                import_libraries("import jax.numpy as jnp") # Use jax.numpy as jnp to avoid namespace issues
                if isinstance(tensor, jnp.ndarray): # Check against jnp.ndarray not jax.Array which is abstract
                    M_np = np.array(tensor) # Convert JAX array to NumPy array
                else:
                    raise TypeError("Input tensor must be a NumPy array, PyTorch Tensor, or JAX Array")
            except ImportError:
                raise TypeError("Input tensor must be a NumPy array, PyTorch Tensor, or JAX Array")


    M_np = np.array(M_np) # Ensure it's a NumPy array for consistent handling

    def format_row(row):
        row = np.round(row, 3)
        row_str_list = [f"{x:.3f}" for x in row]
        return " & ".join(row_str_list)

    def process_dimension(arr):
        if arr.ndim <= 1: # Treat 1D or scalar as a single row
            return format_row(arr) + r" \\" + "\n"
        elif arr.ndim == 2: # Standard 2D matrix
            buffer = ""
            for row in arr:
                buffer += format_row(row)
                buffer += r" \\" + "\n"
            return buffer
        else: # Handle higher dimensions recursively, stacking matrices
            buffer = ""
            for slice_ in arr: # Iterate over the first dimension
                buffer += process_dimension(slice_) # Recursively process sub-dimensions
                buffer += r"\\ \hline" + "\n" # Add horizontal line between "matrices"
            return buffer

    matrix_body = process_dimension(M_np)
    latex_matrix = r"\begin{bmatrix}" + "\n" + matrix_body.rstrip() + r"\end{bmatrix}"
    return latex_matrix


def tensor_to_latex(tensor: Union["torch.Tensor", "jax.Array", np.ndarray], matrix_env: bool = True):
    """
    Converts a PyTorch Tensor, JAX Array, or NumPy array of any dimension
    into a LaTeX string representation.

    Args:
        tensor: The input tensor or array-like object.
        matrix_env: If True (default), returns LaTeX in a 'bmatrix' environment for matrices.
                    If False, returns LaTeX for a single row array (inline math style).

    Returns:
        str: A LaTeX string representing the tensor/array.
             Numbers are rounded to 3 decimal places.
    """
    if matrix_env:
        return tensor_matrix_to_latex(tensor)
    else:
        return tensor_array_to_latex(tensor)



def ProfCallback(
    cpu: bool = True,
    cuda: bool = True,
    profile_memory: bool = DEFAULT_TRACE_OPTS["profile_memory"],
    with_stack: bool = DEFAULT_TRACE_OPTS["with_stack"],
    record_shapes: bool = DEFAULT_TRACE_OPTS["record_shapes"],
    with_flops: bool = DEFAULT_TRACE_OPTS["with_flops"],
    # `torch.profiler.schedule` args - note we defer setting these to enable more fine-grained
    # warnings within this setup function
    wait_steps: Optional[int] = None,
    warmup_steps: Optional[int] = None,
    active_steps: Optional[int] = None,
    num_cycles: Optional[int] = None,
    output_dir: Optional[str] = None,
    export_memory_timeline: bool = True,
    export_stacks: bool = True,
    export_key_averages: bool = True,
    enable_nsight_systems: bool = False,
    nsight_systems_output_file: Optional[str] = None,
):
    """
    Initializes and returns a Hugging Face TrainerCallback for PyTorch profiling.

    This callback leverages `torch.profiler` to collect performance data during training.
    The profiler will be active for a specified number of steps and cycles, as defined by
    the schedule parameters.

    Args:
        cpu (bool): Whether to profile CPU activities. Defaults to True.
        cuda (bool): Whether to profile CUDA activities. Defaults to True.
        profile_memory (bool): Whether to profile memory usage. Enabling this can
            significantly increase profile size. Defaults to `True` (as per `DEFAULT_TRACE_OPTS`).
            Set to `False` to reduce profile size.
        with_stack (bool): Whether to include stack traces in the profile. Stack traces
            can be very verbose. Disabling this can reduce profile size.
            Defaults to `True`. Note that `profile_memory=True` forces this to `True`.
        record_shapes (bool): Whether to record shapes of tensors. For models with many
            tensors, this can increase profile size. Defaults to `True`.
            Note that `profile_memory=True` forces this to `True`.
        with_flops (bool): Whether to report FLOPs. Defaults to `True`.
        wait_steps (Optional[int]): Number of initial steps to ignore before starting
            any profiling activity. If not set, defaults to `DEFAULT_SCHEDULE["wait_steps"]` (4).
        warmup_steps (Optional[int]): Number of steps for profiler warmup after `wait_steps`
            and before active recording. If not set, defaults to `DEFAULT_SCHEDULE["warmup_steps"]` (4).
        active_steps (Optional[int]): Number of steps to actively record profiling data.
            Reducing this value is a key way to decrease profile file size.
            If not set, defaults to `DEFAULT_SCHEDULE["active_steps"]` (1).
        num_cycles (Optional[int]): Number of times the wait, warmup, active sequence is repeated.
            Reducing this limits the number of profiling snapshots taken.
            If not set, defaults to `DEFAULT_SCHEDULE["num_cycles"]` (1).
        output_dir (Optional[str]): Directory to save profiling results.
            Defaults to "profiler_output".
        export_memory_timeline (bool): Whether to export the memory timeline artifact. Defaults to True.
        export_stacks (bool): Whether to export stack trace artifacts. Defaults to True.
        export_key_averages (bool): Whether to export key averages (operator table). Defaults to True.
        enable_nsight_systems (bool): If True, configures the profiler to work with NVIDIA Nsight Systems.
            If the script is detected to be running under `nsys profile` (via `NSYS_PROFILING_SESSION_ID` env var),
            PyTorch's own trace artifact generation via `trace_handler` will be disabled to prevent redundancy,
            relying on Nsight Systems to capture NVTX ranges.
            If not running under `nsys`, a warning will be issued suggesting how to launch with `nsys`,
            and the PyTorch profiler will be disabled. Defaults to False.
        nsight_systems_output_file (Optional[str]): Suggested output file path for Nsight Systems profiling
            when `enable_nsight_systems` is True but the script isn't run under `nsys`.
            If None, a default like "profiler_output/nsys_profile.nsys-rep" is suggested.

    Returns:
        transformers.TrainerCallback: An instance of the profiler callback.

    How to Reduce Profile Size:
    Dealing with large models can produce very large profile files. Here's how to manage this:

    1.  Disable Detailed Tracing Options:
        *   `profile_memory=False`: Disables memory profiling. Memory snapshots are a
            major contributor to profile size. Disabling this offers the most significant reduction.
        *   `with_stack=False`: Excludes Python and CUDA stack traces from the profile.
            Stack traces add considerable verbosity. This is automatically enabled if
            `profile_memory` is true.
        *   `record_shapes=False`: Prevents recording of tensor shapes. While useful for
            debugging, shape information can bloat profiles for models with numerous tensors.
            This is automatically enabled if `profile_memory` is true.

    2.  Adjust Profiling Schedule:
        The profiling schedule dictates when and for how long data is collected.
        *   `wait_steps`: These are steps where the profiler is idle before warmup.
            Does not directly impact size but defines the start of the profiling window.
        *   `warmup_steps`: Steps for the profiler to warm up (e.g., allow JIT compilation
            to complete). Data from these steps is not recorded. Does not directly impact size.
        *   `active_steps`: This is the crucial parameter for size. It's the number of
            steps where data is actively collected. Reducing `active_steps` (e.g., to 1 or 2)
            will directly lead to smaller profile files as less execution time is captured.
        *   `num_cycles`: The number of times the `wait_steps -> warmup_steps -> active_steps`
            sequence is repeated. Reducing `num_cycles` (e.g., to 1) means profiling
            happens for fewer distinct periods in the training run, thus generating less data.
            For example, if you only need a snapshot at a specific point, set `num_cycles=1`.

    By tuning these parameters, especially `active_steps`, `num_cycles`, and `profile_memory`,
    you can significantly reduce the size of the generated profile traces, making them
    more manageable for large-scale model training.
    """
    import_libraries(
        "from transformers import TrainerCallback",
        "import torch",
        "from torch._C._profiler import _ExperimentalConfig",
        "from pathlib import Path",
        "from functools import partial"
    )

    PROFILER_KEY = "profiler"
    DEFAULT_PROFILER_ACTIVITIES = {
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    }

    DEFAULT_SCHEDULE: dict = {
        "wait_steps": 4,
        "warmup_steps": 4,
        "active_steps": 1,
        "num_cycles": 1,
    }


    DEFAULT_PROFILE_DIR: str = "profiler_output"

    class prof_callback(TrainerCallback):
        def __init__(
            self, 
            cpu: bool = True,
            cuda: bool = True,
            profile_memory: bool = DEFAULT_TRACE_OPTS["profile_memory"],
            with_stack: bool = DEFAULT_TRACE_OPTS["with_stack"],
            record_shapes: bool = DEFAULT_TRACE_OPTS["record_shapes"],
            with_flops: bool = DEFAULT_TRACE_OPTS["with_flops"],
            # `torch.profiler.schedule` args - note we defer setting these to enable more fine-grained
            # warnings within this setup function
            wait_steps: Optional[int] = None,
            warmup_steps: Optional[int] = None,
            active_steps: Optional[int] = None,
            num_cycles: Optional[int] = None,
            output_dir: Optional[str] = None,
            export_memory_timeline: bool = True,
            export_stacks: bool = True,
            export_key_averages: bool = True,
            enable_nsight_systems: bool = False,
            nsight_systems_output_file: Optional[str] = None,
        ):
            self.export_memory_timeline = export_memory_timeline
            self.export_stacks = export_stacks
            self.export_key_averages = export_key_averages
            self.enable_nsight_systems = enable_nsight_systems
            self.nsight_systems_output_file = nsight_systems_output_file
            
            self.prof: Optional[torch.profiler.profile] = None # Initialize profiler instance to None

            # Determine base output directory
            self.base_output_dir = output_dir if output_dir is not None else DEFAULT_PROFILE_DIR
            Path(self.base_output_dir).mkdir(parents=True, exist_ok=True) # Ensure it exists

            # Common profiler setup
            activities = []
            if cpu:
                activities.append(torch.profiler.ProfilerActivity.CPU)
            if cuda:
                activities.append(torch.profiler.ProfilerActivity.CUDA)
            if len(activities) == 0:
                print("No activities specified, defaulting to CPU + CUDA")
                activities = DEFAULT_PROFILER_ACTIVITIES
                cpu = cuda = True

            # Check for schedule
            # 1) If no schedule is provided, set to DEFAULT_SCHEDULE
            # 2) else check for missing keys and warn if any are missing, setting these to defaults
            # Note that this might result in code duplication if these checks are already done in the `recipe`
            # However, we retain this checks in the case that the _setup_profiler section of the `recipe` does not implement these checks

            # Set up profiler schedule
            use_default_schedule = not any(
                [
                    wait_steps is not None,
                    warmup_steps is not None,
                    active_steps is not None,
                    num_cycles is not None,
                ]
            )

            # Use default schedule if None, else validate that schedule is valid and can be passed to `instantiate`
            if use_default_schedule:
                schedule_args = DEFAULT_SCHEDULE
                print(
                    " No schedule found in config, defaulting to {}".format(
                        ", ".join(f"{k} = {schedule_args[k]}" for k in schedule_args.keys())
                    )
                )
            else:
                schedule_args = {
                    "wait_steps": wait_steps,
                    "warmup_steps": warmup_steps,
                    "active_steps": active_steps,
                    "num_cycles": num_cycles,
                }
                missing_keys = [k for k in schedule_args.keys() if schedule_args[k] is None]
                if len(missing_keys) > 0:
                    for k in missing_keys:
                        schedule_args[k] = DEFAULT_SCHEDULE[k]
                    print(
                        " Missing keys in torch profiler schedule {}: defaulting to {}".format(
                            ", ".join(missing_keys),
                            ", ".join(f"{k} = {schedule_args[k]}" for k in missing_keys),
                        )
                    )
            schedule = torch.profiler.schedule(
                wait=schedule_args["wait_steps"],
                warmup=schedule_args["warmup_steps"],
                active=schedule_args["active_steps"],
                repeat=schedule_args["num_cycles"],
            )

            # profile_memory requires with_stack and record_shapes, hence we override these if profile_memory is True
            # See torch.profiler.profiler._memory_profile
            if profile_memory:
                print(
                    "`profile_memory` requires `with_stack` and `record_shapes`, these will be enabled since `profile_memory` is True"
                )
            with_stack = with_stack or profile_memory
            record_shapes = record_shapes or profile_memory
            # experimental config is needed to export stacks: see https://github.com/pytorch/pytorch/issues/100253
            experimental_config = _ExperimentalConfig(verbose=True) if with_stack else None

            # Handle exporting of trace, memory timeline and other profiler artifacts
            # output_dir variable here is the one from the outer ProfCallback scope
            effective_output_dir = str(self.base_output_dir)

            on_trace_ready_handler = None
            
            if self.enable_nsight_systems:
                # Check if running under Nsight Systems
                if os.getenv("NSYS_PROFILING_SESSION_ID"):
                    if get_world_size_and_rank()[1] == 0: # Rank 0
                        print(
                            "INFO: Nsight Systems profiling session detected. "
                            "PyTorch NVTX ranges will be captured by Nsight Systems. "
                            "PyTorch's own trace artifact generation via trace_handler will be minimized."
                        )
                    # PyTorch profiler still runs to generate NVTX ranges, but trace_handler is disabled
                    on_trace_ready_handler = None 
                else:
                    # Nsight Systems enabled, but not running under nsys
                    if get_world_size_and_rank()[1] == 0: # Rank 0
                        nsys_output_path = self.nsight_systems_output_file or os.path.join(effective_output_dir, "nsys_profile.nsys-rep")
                        print(
                            f"WARNING: Nsight Systems profiling is enabled via enable_nsight_systems=True, "
                            f"but the script does not appear to be running under 'nsys profile'. "
                            f"The PyTorch profiler will be disabled. \n"
                            f"Please re-launch your script using a command like: \n"
                            f"nsys profile -o {nsys_output_path} python your_script.py <your_args>"
                        )
                    self.prof = None # Disable PyTorch profiler
                    # No need to proceed with profiler setup if it's disabled
                    return 
            
            # If not using Nsight Systems or if Nsight is active (on_trace_ready_handler is None for Nsight)
            if not self.enable_nsight_systems:
                on_trace_ready_handler = partial(
                    trace_handler,
                    output_dir=effective_output_dir,
                    export_memory_timeline=self.export_memory_timeline,
                    export_stacks=self.export_stacks,
                    export_key_averages=self.export_key_averages,
                )

            self.prof = torch.profiler.profile(
                activities=activities,
                profile_memory=profile_memory, # Still respected for NVTX
                with_stack=with_stack,         # Still respected for NVTX
                record_shapes=record_shapes,   # Still respected for NVTX
                with_flops=with_flops,
                schedule=schedule,
                experimental_config=experimental_config,
                on_trace_ready=on_trace_ready_handler, # This is None if nsys is active, else our handler
            )

            self.schedule_args = schedule_args

            self.wait_steps = schedule_args["wait_steps"]
            self.warmup_steps = schedule_args["warmup_steps"]
            self.active_steps = schedule_args["active_steps"]
            self.repeat = schedule_args["num_cycles"]

            self.current_repeat = 0
            self.is_rank_zero = False

        def on_train_begin(self, args , state, control, **kwargs):
            self.is_rank_zero = args.local_rank in [-1, 0]
            if self.prof:
                self.prof.start()

        def on_step_begin(self, args, state, control, **kwargs):
            if not self.is_rank_zero:
                return

            # Memory recording control should only happen if PyTorch profiler is active 
            # and memory profiling is specifically enabled for it.
            # Nsight Systems handles its own memory tracing if configured.
            if self.prof and self.prof.profile_memory: # Check if profiler exists and is set to profile memory
                 print(f"{state.global_step = }")
                 if state.epoch < 1 and state.global_step == self.wait_steps + self.warmup_steps:
                     print("Starting memory recording for PyTorch profiler...")
                     torch.cuda.memory._record_memory_history()

        def on_step_end(self, args, state, control, **kwargs):
            # Memory recording control
            if self.prof and self.prof.profile_memory: # Check if profiler exists and is set to profile memory
                if state.epoch < 1 and state.global_step == (self.wait_steps + self.warmup_steps + self.active_steps + 1):
                    print("Stopping memory recording for PyTorch profiler...")
                    torch.cuda.memory._record_memory_history(enabled=False)
            
            if self.prof:
                self.prof.step()

        def on_train_end(self, args, state, control, **kwargs):
            if self.prof:
                self.prof.stop()

    sig = inspect.signature(ProfCallback)
    passed_args = {
        k: v
        for k, v in locals().items()
        if k in sig.parameters and k != "self"
    }
    return prof_callback(**passed_args)


def create_dynamic_function(source, function_name, target_class=None, original_func=None, filename_prefix="<dynamic>"):
    """
    We assume that we already patch the function, the rest step is to just call 
    `exec(source, globals())`. This is to enable `inspect.getsource` on the patched function.
    """
    frame = inspect.currentframe().f_back
    unique_id = id(frame)
    frame_info = inspect.getframeinfo(frame)
    patch_filepath = os.path.abspath(frame_info.filename)
    patch_line_no = frame_info.lineno

    if original_func:
        original_func_name = original_func.__name__
        filename = f"{filename_prefix}-{original_func_name}-{patch_filepath}-{patch_line_no}-{unique_id}"
    else:
        filename = f"{filename_prefix}-{patch_filepath}-{patch_line_no}-{unique_id}"
    
    # Sanitize filename
    filename = re.sub(r"[^\w\-_\.]", "_", filename)

    code = compile(source, filename, "exec")
    globals_ = frame.f_globals
    locals_ = frame.f_locals

    temp_locals = {}
    exec(code, globals_, temp_locals)
    func = temp_locals[function_name]

    lines = [line + "\n" for line in source.split("\n")]
    linecache.cache[filename] = (
        len(source),
        None,
        lines,
        filename,
    )

    if target_class is not None:
        setattr(target_class, function_name, func)

    return func

def take_subset_and_upload(
    dataset_name_or_path: str,
    output_path: str,
    subset_size: int=10_000,
    eval_subset_size: int=1_000
):
    from datasets import load_dataset, Dataset, DatasetDict

    fineweb_edu_dataset = load_dataset(dataset_name_or_path, streaming=True, split="train")

    train_dataset = list(fineweb_edu_dataset.take(subset_size))
    list_train_dataset = list(train_dataset)
    list_train_dataset = [{"text" : x["text"]} for x in list_train_dataset]

    shuffled_dataset = fineweb_edu_dataset.shuffle(seed=42, buffer_size=10_000)
    eval_dataset = shuffled_dataset.take(eval_subset_size)
    list_eval_dataset = list(eval_dataset)
    list_eval_dataset = [{"text" : x["text"]} for x in list_eval_dataset]

    new_dataset = DatasetDict({"train": Dataset.from_list(list_train_dataset), "dev": Dataset.from_list(list_eval_dataset)})
    new_dataset.push_to_hub(output_path)