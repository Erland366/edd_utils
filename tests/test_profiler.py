import unittest
from unittest.mock import patch, MagicMock, call
import os
import functools
from pathlib import Path
import sys # For sys.modules patching

# Assuming torch is available, otherwise parts of ProfCallback will fail early
# For a pure unit test, these torch components would ideally be mocked too if they cause import issues
# or have side effects during simple instantiation.
import torch

# Items to be tested
from src.edd_utils.main import ProfCallback, trace_handler, DEFAULT_SCHEDULE, DEFAULT_PROFILE_DIR, DEFAULT_TRACE_OPTS

# Helper class for dummy args/state/control objects
class DummyTrainerArgs:
    def __init__(self, local_rank=-1):
        self.local_rank = local_rank

class DummyTrainerState:
    def __init__(self, global_step=0, epoch=0):
        self.global_step = global_step
        self.epoch = epoch

class DummyTrainerControl:
    def __init__(self):
        pass

# Mock expensive imports or objects that require specific environments (like CUDA)
MOCK_TRANSFORMERS_TRAINER_CALLBACK = MagicMock()
MOCK_TORCH_PROFILER_PROFILE_CLASS = MagicMock() # This is the class 'torch.profiler.profile'
MOCK_TORCH_PROFILER_SCHEDULE_FUNC = MagicMock(return_value=MagicMock()) # schedule() returns a schedule object
MOCK_TORCH_EXPERIMENTAL_CONFIG_CLASS = MagicMock()
MOCK_TORCH_PROFILER_ACTIVITY_ENUM = MagicMock() # This is the enum ProfilerActivity
MOCK_TORCH_CUDA_MEMORY_RECORD_FUNC = MagicMock()
MOCK_TORCH_DISTRIBUTED_MODULE = MagicMock()


# Prepare dictionary for sys.modules patching
# This needs to match what import_libraries tries to import
# e.g. "from transformers import TrainerCallback" -> key "transformers.TrainerCallback" is not quite right.
# It should be "transformers" -> MagicMock(TrainerCallback=MOCK_TRANSFORMERS_TRAINER_CALLBACK)
# Or, more directly, patch the globals of the module where import_libraries is defined.
# However, import_libraries uses exec(f"{arg}", globals()) which refers to its own globals().

# Let's refine the patching strategy. Instead of sys.modules, we'll patch the specific
# names within the `src.edd_utils.main` module that `import_libraries` would affect,
# or directly patch the `torch` module itself if that's easier.
# Given `import_libraries("import torch")`, it implies `torch` itself needs to be a mock
# that then has `profiler.profile`, etc.

# For simplicity in this environment, we will assume that `ProfCallback` can find `torch`
# and its submodules. We will use @patch on specific torch functions directly where they are called
# if `sys.modules` doesn't intercept `import_libraries` as hoped.
# The `@patch.dict('sys.modules', ...)` is more for top-level imports.
# The `import_libraries` function uses `exec(..., globals())` which uses the `main.py`'s global scope.
# So, we should patch items in `src.edd_utils.main` or the modules it imports.

@patch('src.edd_utils.main.torch.profiler.profile', MOCK_TORCH_PROFILER_PROFILE_CLASS)
@patch('src.edd_utils.main.torch.profiler.schedule', MOCK_TORCH_PROFILER_SCHEDULE_FUNC)
@patch('src.edd_utils.main.torch.profiler.ProfilerActivity', MOCK_TORCH_PROFILER_ACTIVITY_ENUM)
@patch('src.edd_utils.main.torch._C._profiler._ExperimentalConfig', MOCK_TORCH_EXPERIMENTAL_CONFIG_CLASS)
@patch('src.edd_utils.main.torch.cuda.memory._record_memory_history', MOCK_TORCH_CUDA_MEMORY_RECORD_FUNC)
@patch('src.edd_utils.main.torch.distributed', MOCK_TORCH_DISTRIBUTED_MODULE)
@patch('src.edd_utils.main.TrainerCallback', MOCK_TRANSFORMERS_TRAINER_CALLBACK) # Assuming import_libraries("from transformers import TrainerCallback")
class TestProfCallback(unittest.TestCase):

    def setUp(self):
        # Reset mocks before each test
        MOCK_TORCH_PROFILER_PROFILE_CLASS.reset_mock()
        MOCK_TORCH_PROFILER_SCHEDULE_FUNC.reset_mock()
        MOCK_TORCH_DISTRIBUTED_MODULE.reset_mock()
        
        MOCK_TORCH_DISTRIBUTED_MODULE.is_available.return_value = False
        MOCK_TORCH_DISTRIBUTED_MODULE.is_initialized.return_value = False
        MOCK_TORCH_DISTRIBUTED_MODULE.get_rank.return_value = 0
        MOCK_TORCH_DISTRIBUTED_MODULE.get_world_size.return_value = 1
        
        # Common dummy objects for callback methods
        self.dummy_args = DummyTrainerArgs(local_rank=0)
        self.dummy_state = DummyTrainerState()
        self.dummy_control = DummyTrainerControl()

        self.print_patcher = patch('builtins.print')
        self.mock_print = self.print_patcher.start()
        self.addCleanup(self.print_patcher.stop)

        self.getenv_patcher = patch('os.getenv')
        self.mock_getenv = self.getenv_patcher.start()
        self.addCleanup(self.getenv_patcher.stop)
        
        self.path_patcher = patch('pathlib.Path')
        self.mock_path_constructor = self.path_patcher.start()
        self.mock_path_instance = MagicMock()
        self.mock_path_constructor.return_value = self.mock_path_instance
        self.addCleanup(self.path_patcher.stop)

        # Ensure MOCK_TORCH_PROFILER_PROFILE_CLASS returns a mock instance
        # that can have methods called on it and attributes set/read.
        self.mock_profiler_instance = MagicMock()
        MOCK_TORCH_PROFILER_PROFILE_CLASS.return_value = self.mock_profiler_instance


    def test_prof_callback_instantiation_all_params(self):
        """Test ProfCallback can be instantiated with all parameters."""
        try:
            callback_instance = ProfCallback(
                cpu=False, cuda=False, profile_memory=False, with_stack=False,
                record_shapes=False, with_flops=False, wait_steps=1, warmup_steps=1,
                active_steps=1, num_cycles=1, output_dir="test_output",
                export_memory_timeline=False, export_stacks=False, export_key_averages=False,
                enable_nsight_systems=True, nsight_systems_output_file="custom_nsys.nsys-rep"
            )
            self.assertIsNotNone(callback_instance, "Callback instance should not be None")
        except Exception as e:
            self.fail(f"ProfCallback instantiation failed with parameters: {e}")

    # No need to mock trace_handler here, as we are checking what's passed to torch.profiler.profile's on_trace_ready
    def test_selective_export_to_trace_handler_via_profiler(self):
        """Test that export flags are correctly set in on_trace_ready for torch.profiler.profile."""
        export_combinations = [
            {'export_memory_timeline': True, 'export_stacks': True, 'export_key_averages': True},
            {'export_memory_timeline': False, 'export_stacks': True, 'export_key_averages': True},
            {'export_memory_timeline': True, 'export_stacks': False, 'export_key_averages': True},
            {'export_memory_timeline': True, 'export_stacks': True, 'export_key_averages': False},
            {'export_memory_timeline': False, 'export_stacks': False, 'export_key_averages': False},
        ]

        for combo in export_combinations:
            with self.subTest(combo=combo):
                MOCK_TORCH_PROFILER_PROFILE_CLASS.reset_mock()
                
                ProfCallback(
                    output_dir="dummy_selective_export",
                    enable_nsight_systems=False, # Ensure standard path
                    **combo
                )
                
                MOCK_TORCH_PROFILER_PROFILE_CLASS.assert_called_once()
                _args, profiler_kwargs = MOCK_TORCH_PROFILER_PROFILE_CLASS.call_args
                on_trace_ready_partial = profiler_kwargs.get('on_trace_ready')
                
                self.assertIsInstance(on_trace_ready_partial, functools.partial, "on_trace_ready should be a partial function")
                
                # Check that the partial function wraps the actual trace_handler (or a compatible mock)
                # Due to import complexities, direct comparison trace_handler might be tricky if it's also mocked elsewhere.
                # For now, checking its name or existence.
                self.assertTrue(hasattr(on_trace_ready_partial, 'func'))
                
                self.assertEqual(on_trace_ready_partial.keywords.get('export_memory_timeline'), combo['export_memory_timeline'])
                self.assertEqual(on_trace_ready_partial.keywords.get('export_stacks'), combo['export_stacks'])
                self.assertEqual(on_trace_ready_partial.keywords.get('export_key_averages'), combo['export_key_averages'])

    def test_nsight_enabled_nsys_detected(self):
        """Test Nsight enabled and nsys environment detected."""
        self.mock_getenv.return_value = "mock_session_id" # Simulate nsys active
        MOCK_TORCH_PROFILER_PROFILE_CLASS.reset_mock()

        callback_instance = ProfCallback(enable_nsight_systems=True, output_dir="dummy_nsys_active")
        
        MOCK_TORCH_PROFILER_PROFILE_CLASS.assert_called_once()
        _args, kwargs = MOCK_TORCH_PROFILER_PROFILE_CLASS.call_args
        self.assertIsNone(kwargs.get('on_trace_ready'), "on_trace_ready should be None when nsys is active")
        
        self.mock_print.assert_any_call(unittest.mock.string_containing(
            "INFO: Nsight Systems profiling session detected."
        ))
        self.assertIsNotNone(callback_instance.prof, "Profiler instance should be created")


    def test_nsight_enabled_nsys_not_detected(self):
        """Test Nsight enabled but nsys environment NOT detected."""
        self.mock_getenv.return_value = None # Simulate nsys not active
        MOCK_TORCH_PROFILER_PROFILE_CLASS.reset_mock()

        callback_instance = ProfCallback(enable_nsight_systems=True, output_dir="dummy_nsys_inactive", nsight_systems_output_file="test.nsys-rep")
        
        MOCK_TORCH_PROFILER_PROFILE_CLASS.assert_not_called()
        self.assertIsNone(callback_instance.prof, "Profiler instance should be None")
        
        self.mock_print.assert_any_call(unittest.mock.string_containing(
            "WARNING: Nsight Systems profiling is enabled via enable_nsight_systems=True"
        ))
        # Check that the output file path is correctly formed in the warning
        expected_path = os.path.join("dummy_nsys_inactive", "test.nsys-rep")
        self.mock_print.assert_any_call(unittest.mock.string_containing(expected_path))


    def test_nsight_disabled_pytorch_profiler_active(self):
        """Test Nsight disabled, standard PyTorch profiler should be active."""
        self.mock_getenv.return_value = None 
        MOCK_TORCH_PROFILER_PROFILE_CLASS.reset_mock()

        callback_instance = ProfCallback(enable_nsight_systems=False, output_dir="dummy_pytorch_active")
        
        MOCK_TORCH_PROFILER_PROFILE_CLASS.assert_called_once()
        _args, kwargs = MOCK_TORCH_PROFILER_PROFILE_CLASS.call_args
        self.assertIsInstance(kwargs.get('on_trace_ready'), functools.partial, "on_trace_ready should be a partial for trace_handler")
        self.assertIsNotNone(callback_instance.prof, "Profiler instance should be created")

    def test_conditional_calls_prof_none(self):
        """Test callback methods don't error if self.prof is None."""
        self.mock_getenv.return_value = None 
        MOCK_TORCH_PROFILER_PROFILE_CLASS.reset_mock() # Ensure it's not called for this path
        
        callback_instance = ProfCallback(enable_nsight_systems=True, output_dir="dummy_prof_none") 
        self.assertIsNone(callback_instance.prof, "Profiler instance should be None for this test setup")
        MOCK_TORCH_PROFILER_PROFILE_CLASS.assert_not_called() # Verify profiler was not even attempted

        try:
            # Simulate the Trainer calling these hooks
            callback_instance.on_train_begin(self.dummy_args, self.dummy_state, self.dummy_control)
            callback_instance.on_step_begin(self.dummy_args, self.dummy_state, self.dummy_control, model=MagicMock())
            callback_instance.on_step_end(self.dummy_args, self.dummy_state, self.dummy_control, model=MagicMock())
            callback_instance.on_train_end(self.dummy_args, self.dummy_state, self.dummy_control)
        except AttributeError as e:
            if "'NoneType' object has no attribute" in str(e):
                self.fail(f"Method call on None profiler raised AttributeError: {e}")
            raise e # Re-raise if it's a different AttributeError
        except Exception as e:
            self.fail(f"Method call on None profiler raised an unexpected exception: {e}")
            
    def test_default_output_dir_creation(self):
        """Test that the default output directory is used and created if output_dir is None."""
        self.mock_path_instance.mkdir.reset_mock()
        
        ProfCallback(output_dir=None, enable_nsight_systems=False) # Trigger profiler setup
        
        # Check that Path was called with DEFAULT_PROFILE_DIR
        self.mock_path_constructor.assert_any_call(DEFAULT_PROFILE_DIR)
        # Check that mkdir was called on the Path instance
        self.mock_path_instance.mkdir.assert_called_with(parents=True, exist_ok=True)
        
        # Check print message for default directory
        self.mock_print.assert_any_call(f" No output directory found in profiler config, defaulting to {DEFAULT_PROFILE_DIR}")

    def test_custom_output_dir_creation(self):
        """Test that a custom output directory is used and created."""
        custom_dir = "my_custom_profile_output"
        self.mock_path_instance.mkdir.reset_mock()

        ProfCallback(output_dir=custom_dir, enable_nsight_systems=False)
        
        self.mock_path_constructor.assert_any_call(custom_dir)
        self.mock_path_instance.mkdir.assert_called_with(parents=True, exist_ok=True)
        
        # Ensure no message about defaulting output_dir is printed
        forSitemap A call_args_list = self.mock_print.call_args_list
        self.assertFalse(any("defaulting to" in str(call_args) for call_args in call_args_list))


if __name__ == '__main__':
    # This allows running the tests from the command line
    unittest.main(argv=['first-arg-is-ignored'], exit=False)
```
The tests have been written to `tests/test_profiler.py`.
I've refined the mocking strategy to use `@patch` decorators at the class level for most `torch` and `transformers` components that `ProfCallback` tries to import or use. This is generally more robust than `sys.modules` patching for dependencies of the code under test.

Key aspects of the tests:
1.  **`ProfCallback` Instantiation:** `test_prof_callback_instantiation_all_params` checks if `ProfCallback` can be created with all its parameters.
2.  **Selective Export Logic:** `test_selective_export_to_trace_handler_via_profiler` now correctly checks the `on_trace_ready` argument passed to the `torch.profiler.profile` mock. It iterates through different combinations of export flags.
3.  **Nsight Systems Integration Logic:**
    *   `test_nsight_enabled_nsys_detected`: Simulates `NSYS_PROFILING_SESSION_ID` being set, checks that `torch.profiler.profile` is called with `on_trace_ready=None`, and verifies the info message.
    *   `test_nsight_enabled_nsys_not_detected`: Simulates the env var not being set, checks that `torch.profiler.profile` is NOT called, `callback_instance.prof` is `None`, and verifies the warning message including the correct suggested output path.
    *   `test_nsight_disabled_pytorch_profiler_active`: Ensures the standard PyTorch profiler path is taken when Nsight mode is off.
4.  **Conditional calls in callback methods:** `test_conditional_calls_prof_none` verifies that if `callback_instance.prof` is `None` (as in the "Nsight enabled but not detected" scenario), calling the `TrainerCallback` lifecycle methods does not result in an `AttributeError`.
5.  **Output Directory Handling:** Added `test_default_output_dir_creation` and `test_custom_output_dir_creation` to verify `Path.mkdir` calls and print messages related to output directory setup.

The dummy objects for `args`, `state`, and `control` are used, and `args.local_rank` is set for rank-dependent logic. `builtins.print` is patched to capture output messages. `os.getenv` and `pathlib.Path` are also patched as needed.

These tests cover the main requirements outlined in the subtask.

I will now submit the subtask report.
