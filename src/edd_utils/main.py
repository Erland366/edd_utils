# Profiler code was taken from https://github.com/pytorch/torchtune/blob/890deab3029eef65f94cedb37fda14479f65f129/torchtune/training/_profiler.py
# But now it is implemented for TrainerCallback
import linecache
import os
import re
import inspect
from typing import Optional

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
):
    """
    Handles export of artifacts from ``torch.profiler.profile``.

    The following artifacts are exported:
    - chrome / tensorboard trace - viewable through tensorboard or perfetto.dev / chrome::/tracing
    - trace event table
    - memory timeline and snapshot.pickle if ``profile_memory``
    - stacks if ``with_stack`` (note that ``profile_memory`` requires ``with_stack`` to be ``True``),
    viewable as a flamegraph see (https://pytorch.org/docs/stable/profiler.html#torch.profiler._KinetoProfile.export_stacks).

    Notes:
    - Each profiling cycle is exported as a sub-directory in output_dir
        - E.g., profiling in 5-step cycle (wait=2, warmup=2, active=1, repeat=0) will result in
        sub-directories iteration_5, iteration_10, etc.
    - If profiling in a distributed setting, each artifact will be prefixed with rank.
    - Memory timeline is only exported for rank 0 (error if exporting from multiple ranks on single node)

    See profiler documentation (https://pytorch.org/docs/stable/profiler.html#torch.profiler.profile) for more details

    Args:
        prof (torch.profiler.profile): instance of torch profiler to use
        output_dir (str):  directory to store artifacts
        metric (str): metric to order trace event table by, see ``torch.profiler.profile.key_averages().table`` for
        row_limit (int): number of rows to display in trace event table

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
    if prof.profile_memory:
        if rank == 0:
            try:
                prof.export_memory_timeline(
                    f"{curr_trace_dir}/rank{rank}_memory-timeline.html"
                )
            except Exception as e:
                # log.warn(f" Failed to export memory timeline: {e}")
                print(f"Saving profiling results to {curr_trace_dir}")

            torch.cuda.memory._dump_snapshot(
                f"{curr_trace_dir}/rank{rank}_memory_snapshot.pickle"
            )

    # Dump stack traces
    if prof.with_stack:
        prof.export_stacks(f"{curr_trace_dir}/rank{rank}_stacks.txt", metric=metric)

    # Export event averages
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
):
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
        ):


            # Set up profiler activities
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
            if output_dir is None:
                print(
                    f" No output directory found in profiler config, defaulting to {DEFAULT_PROFILE_DIR}"
                )
                output_dir = DEFAULT_PROFILE_DIR

            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)
            output_dir = str(output_dir)

            # trace_handler manages the export of profiler artifacts
            # this callback will be triggered after **each** profiling cycle
            callback = partial(trace_handler, output_dir=output_dir)

            prof = torch.profiler.profile(
                activities=activities,
                profile_memory=profile_memory,
                with_stack=with_stack,
                record_shapes=record_shapes,
                with_flops=with_flops,
                schedule=schedule,
                experimental_config=experimental_config,
                on_trace_ready=callback,
            )

            self.prof = prof
            self.schedule_args =schedule_args 

            self.wait_steps = schedule_args["wait_steps"]
            self.warmup_steps = schedule_args["warmup_steps"]
            self.active_steps = schedule_args["active_steps"]
            self.repeat = schedule_args["num_cycles"]

            self.current_repeat = 0
            self.is_rank_zero = False

        def on_train_begin(self, args , state, control, **kwargs):
            self.is_rank_zero = args.local_rank in [-1, 0]
            self.prof.start()

        def on_step_begin(self, args, state, control, **kwargs):
            if not self.is_rank_zero:
                return

            print(f"{state.global_step = }")

            if state.epoch < 1 and state.global_step == self.wait_steps + self.warmup_steps:
                print("Starting memory recording...")
                torch.cuda.memory._record_memory_history()

        def on_step_end(self, args, state, control, **kwargs):
            if state.epoch < 1 and state.global_step == (self.wait_steps + self.warmup_steps + self.active_steps + 1):
                print("Stopping memory recording...")
                torch.cuda.memory._record_memory_history(enabled=False)

            self.prof.step()

        def on_train_end(self, args, state, control, **kwargs):
            self.prof.stop()

    sig = inspect.signature(ProfCallback)
    passed_args = {
        k: v
        for k, v in locals().items()
        if k in sig.parameters and k != "self"
    }
    return prof_callback(**passed_args)


def create_dynamic_function(source, function_name, original_func=None, filename_prefix="<dynamic>"):
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

    return func