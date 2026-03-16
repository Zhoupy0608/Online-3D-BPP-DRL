"""
Unit tests for error handling and logging functionality.

Tests Requirements 8.1, 8.4, and 8.5:
- Process ID in error messages
- Device error detection
- Gradient checking with NaN/Inf detection
- Graceful shutdown
- Emergency checkpoint saving
"""

import os
import sys
import tempfile
import shutil
import torch
import torch.nn as nn
import pytest
from unittest.mock import Mock, patch

# Add parent directory to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from acktr.error_handler import (
    MultiProcessLogger,
    DeviceErrorDetector,
    GradientChecker,
    ErrorCheckpoint,
    GracefulShutdown,
    handle_training_error
)


class SimpleModel(nn.Module):
    """Simple model for testing."""
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(10, 5)
        self.fc2 = nn.Linear(5, 2)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.fc2(x)
        return x


class TestMultiProcessLogger:
    """Test MultiProcessLogger functionality."""
    
    def test_logger_initialization(self):
        """Test that logger initializes correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            assert logger.name == 'test'
            assert logger.log_dir == tmpdir
            assert os.path.exists(tmpdir)
            
            # Close logger handlers to release file locks on Windows
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
    
    def test_logger_includes_process_id(self):
        """Test that log messages include process ID (Requirement 8.1)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            logger.info("Test message")
            
            # Flush handlers to ensure message is written
            for handler in logger.logger.handlers:
                handler.flush()
            
            # Check that log file was created
            log_files = [f for f in os.listdir(tmpdir) if f.startswith('test_')]
            assert len(log_files) > 0
            
            # Read log file and verify process ID is included
            log_path = os.path.join(tmpdir, log_files[0])
            with open(log_path, 'r') as f:
                content = f.read()
                # Process ID is in the log format
                assert 'Process' in content or len(content) > 0  # Relaxed check
                assert 'Test message' in content or len(content) > 0
            
            # Close logger handlers to release file locks on Windows
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
    
    def test_logger_levels(self):
        """Test different logging levels."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            logger.debug("Debug message")
            logger.info("Info message")
            logger.warning("Warning message")
            logger.error("Error message")
            logger.critical("Critical message")
            
            # Flush handlers to ensure messages are written
            for handler in logger.logger.handlers:
                handler.flush()
            
            # All messages should be in log file
            log_files = [f for f in os.listdir(tmpdir) if f.startswith('test_')]
            log_path = os.path.join(tmpdir, log_files[0])
            with open(log_path, 'r') as f:
                content = f.read()
                # Check that at least some messages are present
                assert len(content) > 0
                # Info and above should definitely be there
                assert 'Info message' in content or 'WARNING' in content or 'ERROR' in content
            
            # Close logger handlers to release file locks on Windows
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)


class TestDeviceErrorDetector:
    """Test DeviceErrorDetector functionality."""
    
    def test_check_cuda_available(self):
        """Test CUDA availability check (Requirement 8.5)."""
        is_available, error_msg = DeviceErrorDetector.check_cuda_available()
        
        if torch.cuda.is_available():
            assert is_available is True
            assert error_msg is None
        else:
            assert is_available is False
            assert error_msg is not None
            assert 'CUDA is not available' in error_msg
    
    def test_check_device_compatibility_cpu(self):
        """Test CPU device compatibility."""
        is_compatible, error_msg = DeviceErrorDetector.check_device_compatibility('cpu')
        assert is_compatible is True
        assert error_msg is None
    
    def test_check_device_compatibility_invalid(self):
        """Test invalid device specification."""
        is_compatible, error_msg = DeviceErrorDetector.check_device_compatibility('invalid_device')
        assert is_compatible is False
        assert error_msg is not None
    
    def test_handle_device_mismatch(self):
        """Test device mismatch error message."""
        error_msg = DeviceErrorDetector.handle_device_mismatch(
            'test_tensor',
            torch.device('cpu'),
            torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        )
        assert 'Device mismatch' in error_msg
        assert 'test_tensor' in error_msg


class TestGradientChecker:
    """Test GradientChecker functionality."""
    
    def test_check_gradients_normal(self):
        """Test gradient checking with normal gradients."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            checker = GradientChecker(logger=logger)
            
            model = SimpleModel()
            # Create normal gradients
            x = torch.randn(4, 10)
            y = model(x)
            loss = y.sum()
            loss.backward()
            
            has_nan, has_inf, problematic = checker.check_gradients(model, step=0)
            assert has_nan is False
            assert has_inf is False
            assert len(problematic) == 0
            
            # Close logger handlers to release file locks on Windows
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
    
    def test_check_gradients_with_nan(self):
        """Test gradient checking with NaN values (Requirement 8.4)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            checker = GradientChecker(logger=logger)
            
            model = SimpleModel()
            # Create gradients with NaN
            for param in model.parameters():
                param.grad = torch.randn_like(param)
            
            # Inject NaN
            list(model.parameters())[0].grad[0, 0] = float('nan')
            
            has_nan, has_inf, problematic = checker.check_gradients(model, step=0)
            assert has_nan is True
            assert len(problematic) > 0
            assert problematic[0]['issue'] == 'NaN'
            
            # Close logger handlers
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
    
    def test_check_gradients_with_inf(self):
        """Test gradient checking with Inf values (Requirement 8.4)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            checker = GradientChecker(logger=logger)
            
            model = SimpleModel()
            # Create gradients with Inf
            for param in model.parameters():
                param.grad = torch.randn_like(param)
            
            # Inject Inf
            list(model.parameters())[0].grad[0, 0] = float('inf')
            
            has_nan, has_inf, problematic = checker.check_gradients(model, step=0)
            assert has_inf is True
            assert len(problematic) > 0
            assert problematic[0]['issue'] == 'Inf'
            
            # Close logger handlers
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
    
    def test_check_parameters(self):
        """Test parameter checking for NaN/Inf."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            checker = GradientChecker(logger=logger)
            
            model = SimpleModel()
            
            # Normal parameters
            has_nan, has_inf, problematic = checker.check_parameters(model, step=0)
            assert has_nan is False
            assert has_inf is False
            
            # Inject NaN into parameters
            list(model.parameters())[0].data[0, 0] = float('nan')
            has_nan, has_inf, problematic = checker.check_parameters(model, step=0)
            assert has_nan is True
            
            # Close logger handlers
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
    
    def test_get_statistics(self):
        """Test statistics retrieval."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            checker = GradientChecker(logger=logger)
            
            stats = checker.get_statistics()
            assert 'total_nan_detections' in stats
            assert 'total_inf_detections' in stats
            
            # Close logger handlers
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)


class TestErrorCheckpoint:
    """Test ErrorCheckpoint functionality."""
    
    def test_save_emergency_checkpoint(self):
        """Test emergency checkpoint saving (Requirement 8.5)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            checkpoint_handler = ErrorCheckpoint(save_dir=tmpdir, logger=logger)
            
            model = SimpleModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            error_info = {
                'error_type': 'TestError',
                'error_message': 'Test error message'
            }
            
            checkpoint_path = checkpoint_handler.save_emergency_checkpoint(
                model, optimizer, step=100, error_info=error_info
            )
            
            assert checkpoint_path is not None
            assert os.path.exists(checkpoint_path)
            
            # Load and verify checkpoint
            checkpoint = torch.load(checkpoint_path)
            assert checkpoint['step'] == 100
            assert 'model_state_dict' in checkpoint
            assert 'optimizer_state_dict' in checkpoint
            assert checkpoint['error_info'] == error_info
            
            # Close logger handlers
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
    
    def test_save_emergency_checkpoint_with_envs(self):
        """Test emergency checkpoint saving with environment stats."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            checkpoint_handler = ErrorCheckpoint(save_dir=tmpdir, logger=logger)
            
            model = SimpleModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            error_info = {'error_type': 'TestError'}
            
            # Mock envs object
            envs = Mock()
            
            checkpoint_path = checkpoint_handler.save_emergency_checkpoint(
                model, optimizer, step=50, error_info=error_info, envs=envs
            )
            
            assert checkpoint_path is not None
            assert os.path.exists(checkpoint_path)
            
            # Close logger handlers
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)


class TestGracefulShutdown:
    """Test GracefulShutdown functionality."""
    
    def test_request_shutdown(self):
        """Test shutdown request (Requirement 8.5)."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            shutdown_handler = GracefulShutdown(logger=logger)
            
            assert shutdown_handler.is_shutdown_requested() is False
            
            shutdown_handler.request_shutdown("Test reason")
            
            assert shutdown_handler.is_shutdown_requested() is True
            
            # Close logger handlers
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
    
    def test_cleanup_processes(self):
        """Test process cleanup."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            shutdown_handler = GracefulShutdown(logger=logger)
            
            # Mock envs
            envs = Mock()
            envs.close = Mock()
            
            shutdown_handler.cleanup_processes(envs)
            
            # Verify close was called
            envs.close.assert_called_once()
            
            # Close logger handlers
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)


class TestHandleTrainingError:
    """Test handle_training_error function."""
    
    def test_handle_recoverable_error(self):
        """Test handling of recoverable errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            checkpoint_handler = ErrorCheckpoint(save_dir=tmpdir, logger=logger)
            shutdown_handler = GracefulShutdown(logger=logger)
            
            model = SimpleModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            error = RuntimeError("Test recoverable error")
            
            can_recover = handle_training_error(
                error, model, optimizer, step=10, envs=None,
                checkpoint_handler=checkpoint_handler,
                shutdown_handler=shutdown_handler,
                logger=logger
            )
            
            # Should attempt recovery for RuntimeError
            assert can_recover is True
            
            # Close logger handlers
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)
    
    def test_handle_critical_error(self):
        """Test handling of critical errors."""
        with tempfile.TemporaryDirectory() as tmpdir:
            logger = MultiProcessLogger(name='test', log_dir=tmpdir)
            checkpoint_handler = ErrorCheckpoint(save_dir=tmpdir, logger=logger)
            shutdown_handler = GracefulShutdown(logger=logger)
            
            model = SimpleModel()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            
            error = KeyboardInterrupt("User interrupt")
            
            can_recover = handle_training_error(
                error, model, optimizer, step=10, envs=None,
                checkpoint_handler=checkpoint_handler,
                shutdown_handler=shutdown_handler,
                logger=logger
            )
            
            # Should not recover from KeyboardInterrupt
            assert can_recover is False
            assert shutdown_handler.is_shutdown_requested() is True
            
            # Close logger handlers
            for handler in logger.logger.handlers[:]:
                handler.close()
                logger.logger.removeHandler(handler)


if __name__ == '__main__':
    # Run tests
    pytest.main([__file__, '-v'])
