import linecache
import os
import re
import inspect

def import_libraries(*args):
    for arg in args:
        exec(f"{arg}", globals())

def test_import():
    import_libraries("import numpy as np")

    print(np.array([1, 2, 3, 4, 5]))

def ProfCallback():
    import_libraries("from transformers import TrainerCallback", "import torch")

    class prof_callback(TrainerCallback):
        def __init__(self, prof, profiler_schedule):
            self.prof = prof
            self.profiler_schedule = profiler_schedule

            self.wait_steps = profiler_schedule["wait_steps"]
            self.warmup_steps = profiler_schedule["warmup_steps"]
            self.active_steps = profiler_schedule["active_steps"]
            self.repeat = profiler_schedule["num_cycles"]

            self.current_repeat = 0
            self.is_rank_zero = False

        def on_train_begin(self, args , state, control, **kwargs):
            self.is_rank_zero = args.local_rank in [-1, 0]

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

    return prof_callback


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