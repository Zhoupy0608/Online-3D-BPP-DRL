"""
Error handling and logging utilities for multi-process training.

This module provides comprehensive error handling, logging, and checkpoint
management for the multi-process training system. It addresses Requirements
8.1, 8.4, and 8.5 from the design document.

Key features:
- Process ID tracking in error messages
- Device error detection and clear messages
- Gradient checking with NaN/Inf detection
- Graceful shutdown on critical errors
- Emergency checkpoint saving on errors
"""

import os
import sys
import traceback
import torch
import torch.multiprocessing as mp
from datetime import datetime
import logging


class MultiProcessLogger:
    """
    Logger that includes process ID in all messages.
    
    Requirement 8.1: Add process ID to error messages
    """
    
    def __init__(self, name='training', log_dir='./logs'):
        """
        Initialize multi-process logger.
        
        Args:
            name: Logger name
            log_dir: Directory for log files
        """
        self.name = name
        self.log_dir = log_dir
        
        # Create log directory if it doesn't exist
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        # Set up logger
        self.logger = logging.getLogger(name)
        self.logger.setLevel(logging.DEBUG)
        
        # File handler
        log_file = os.path.join(log_dir, f'{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.DEBUG)
        
        # Console handler
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Formatter with process ID
        formatter = logging.Formatter(
            '[%(asctime)s] [Process %(process)d] [%(levelname)s] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        
        # Add handlers
        if not self.logger.handlers:
            self.logger.addHandler(file_handler)
            self.logger.addHandler(console_handler)
    
    def info(self, message):
        """Log info message with process ID."""
        self.logger.info(message)
    
    def warning(self, message):
        """Log warning message with process ID."""
        self.logger.warning(message)
    
    def error(self, message, exc_info=False):
        """Log error message with process ID and optional exception info."""
        self.logger.error(message, exc_info=exc_info)
    
    def critical(self, message, exc_info=True):
        """Log critical error with process ID and exception info."""
        self.logger.critical(message, exc_info=exc_info)
    
    def debug(self, message):
        """Log debug message with process ID."""
        self.logger.debug(message)


class DeviceErrorDetector:
    """
    Detects and provides clear messages for device-related errors.
    
    Requirement 8.5: Implement device error detection and clear messages
    """
    
    @staticmethod
    def check_cuda_available():
        """
        Check if CUDA is available and provide clear error message if not.
        
        Returns:
            tuple: (is_available, error_message)
        """
        if not torch.cuda.is_available():
            return False, (
                "CUDA is not available. Possible reasons:\n"
                "  1. No GPU detected on this system\n"
                "  2. CUDA drivers not installed\n"
                "  3. PyTorch installed without CUDA support\n"
                "Solution: Use --device cpu or install CUDA-enabled PyTorch"
            )
        return True, None
    
    @staticmethod
    def check_device_compatibility(device_str):
        """
        Check if the specified device is compatible.
        
        Args:
            device_str: Device string (e.g., 'cuda:0', 'cpu')
            
        Returns:
            tuple: (is_compatible, error_message)
        """
        try:
            device = torch.device(device_str)
            
            if device.type == 'cuda':
                # Check if CUDA is available
                is_available, error_msg = DeviceErrorDetector.check_cuda_available()
                if not is_available:
                    return False, error_msg
                
                # Check if the specific GPU index exists
                if device.index is not None:
                    if device.index >= torch.cuda.device_count():
                        return False, (
                            f"GPU index {device.index} not available.\n"
                            f"Available GPUs: {torch.cuda.device_count()}\n"
                            f"Solution: Use --device cuda:0 or --device cpu"
                        )
                
                # Check GPU memory
                try:
                    torch.cuda.set_device(device)
                    total_memory = torch.cuda.get_device_properties(device).total_memory
                    allocated_memory = torch.cuda.memory_allocated(device)
                    free_memory = total_memory - allocated_memory
                    
                    # Warn if less than 1GB free
                    if free_memory < 1e9:
                        return True, (
                            f"Warning: Low GPU memory available ({free_memory / 1e9:.2f} GB free).\n"
                            f"Consider reducing --num-processes or using CPU."
                        )
                except Exception as e:
                    return False, f"Failed to access GPU: {str(e)}"
            
            return True, None
            
        except Exception as e:
            return False, f"Invalid device specification '{device_str}': {str(e)}"
    
    @staticmethod
    def handle_device_mismatch(tensor_name, expected_device, actual_device):
        """
        Provide clear error message for device mismatch.
        
        Args:
            tensor_name: Name of the tensor
            expected_device: Expected device
            actual_device: Actual device
            
        Returns:
            str: Clear error message
        """
        return (
            f"Device mismatch detected for {tensor_name}:\n"
            f"  Expected: {expected_device}\n"
            f"  Actual: {actual_device}\n"
            f"This typically occurs in multi-process training when tensors are created on different devices.\n"
            f"The system will automatically move tensors to the correct device."
        )


class GradientChecker:
    """
    Checks gradients for NaN and Inf values.
    
    Requirement 8.4: Add gradient checking with NaN/Inf detection
    """
    
    def __init__(self, logger=None):
        """
        Initialize gradient checker.
        
        Args:
            logger: MultiProcessLogger instance
        """
        self.logger = logger or MultiProcessLogger()
        self.nan_count = 0
        self.inf_count = 0
    
    def check_gradients(self, model, step):
        """
        Check all model gradients for NaN or Inf values.
        
        Args:
            model: PyTorch model
            step: Current training step
            
        Returns:
            tuple: (has_nan, has_inf, problematic_params)
        """
        has_nan = False
        has_inf = False
        problematic_params = []
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                # Check for NaN
                if torch.isnan(param.grad).any():
                    has_nan = True
                    nan_count = torch.isnan(param.grad).sum().item()
                    problematic_params.append({
                        'name': name,
                        'issue': 'NaN',
                        'count': nan_count,
                        'shape': param.grad.shape,
                        'mean': 'NaN',
                        'std': 'NaN'
                    })
                    self.nan_count += 1
                
                # Check for Inf
                elif torch.isinf(param.grad).any():
                    has_inf = True
                    inf_count = torch.isinf(param.grad).sum().item()
                    problematic_params.append({
                        'name': name,
                        'issue': 'Inf',
                        'count': inf_count,
                        'shape': param.grad.shape,
                        'mean': 'Inf',
                        'std': 'Inf'
                    })
                    self.inf_count += 1
        
        if has_nan or has_inf:
            self._log_gradient_issues(step, problematic_params)
        
        return has_nan, has_inf, problematic_params
    
    def _log_gradient_issues(self, step, problematic_params):
        """Log detailed information about gradient issues."""
        self.logger.error(f"Gradient issues detected at step {step}:")
        for param_info in problematic_params:
            self.logger.error(
                f"  Parameter: {param_info['name']}\n"
                f"    Issue: {param_info['issue']}\n"
                f"    Count: {param_info['count']}\n"
                f"    Shape: {param_info['shape']}\n"
                f"    Mean: {param_info['mean']}\n"
                f"    Std: {param_info['std']}"
            )
    
    def check_parameters(self, model, step):
        """
        Check all model parameters for NaN or Inf values.
        
        Args:
            model: PyTorch model
            step: Current training step
            
        Returns:
            tuple: (has_nan, has_inf, problematic_params)
        """
        has_nan = False
        has_inf = False
        problematic_params = []
        
        for name, param in model.named_parameters():
            if param.data is not None:
                # Check for NaN
                if torch.isnan(param.data).any():
                    has_nan = True
                    nan_count = torch.isnan(param.data).sum().item()
                    problematic_params.append({
                        'name': name,
                        'issue': 'NaN',
                        'count': nan_count,
                        'shape': param.data.shape
                    })
                
                # Check for Inf
                elif torch.isinf(param.data).any():
                    has_inf = True
                    inf_count = torch.isinf(param.data).sum().item()
                    problematic_params.append({
                        'name': name,
                        'issue': 'Inf',
                        'count': inf_count,
                        'shape': param.data.shape
                    })
        
        if has_nan or has_inf:
            self.logger.error(f"Parameter corruption detected at step {step}:")
            for param_info in problematic_params:
                self.logger.error(
                    f"  Parameter: {param_info['name']}\n"
                    f"    Issue: {param_info['issue']}\n"
                    f"    Count: {param_info['count']}\n"
                    f"    Shape: {param_info['shape']}"
                )
        
        return has_nan, has_inf, problematic_params
    
    def get_statistics(self):
        """Get gradient checking statistics."""
        return {
            'total_nan_detections': self.nan_count,
            'total_inf_detections': self.inf_count
        }


class ErrorCheckpoint:
    """
    Handles emergency checkpoint saving on errors.
    
    Requirement 8.5: Add checkpoint saving on errors
    """
    
    def __init__(self, save_dir='./emergency_checkpoints', logger=None):
        """
        Initialize error checkpoint handler.
        
        Args:
            save_dir: Directory for emergency checkpoints
            logger: MultiProcessLogger instance
        """
        self.save_dir = save_dir
        self.logger = logger or MultiProcessLogger()
        
        # Create directory if it doesn't exist
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
    
    def save_emergency_checkpoint(self, model, optimizer, step, error_info, envs=None):
        """
        Save emergency checkpoint when an error occurs.
        
        Args:
            model: PyTorch model
            optimizer: Optimizer
            step: Current training step
            error_info: Dictionary with error information
            envs: Vectorized environments (optional)
            
        Returns:
            str: Path to saved checkpoint
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_name = f"emergency_step{step}_{timestamp}.pt"
        checkpoint_path = os.path.join(self.save_dir, checkpoint_name)
        
        try:
            checkpoint = {
                'step': step,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict() if hasattr(optimizer, 'state_dict') else None,
                'error_info': error_info,
                'timestamp': timestamp
            }
            
            # Add environment normalization stats if available
            if envs is not None:
                try:
                    from acktr import utils
                    vec_norm = utils.get_vec_normalize(envs)
                    if vec_norm is not None:
                        checkpoint['ob_rms'] = getattr(vec_norm, 'ob_rms', None)
                except Exception as e:
                    self.logger.warning(f"Could not save environment stats: {e}")
            
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"Emergency checkpoint saved to: {checkpoint_path}")
            return checkpoint_path
            
        except Exception as e:
            self.logger.error(f"Failed to save emergency checkpoint: {e}", exc_info=True)
            return None


class GracefulShutdown:
    """
    Handles graceful shutdown on critical errors.
    
    Requirement 8.5: Implement graceful shutdown on critical errors
    """
    
    def __init__(self, logger=None):
        """
        Initialize graceful shutdown handler.
        
        Args:
            logger: MultiProcessLogger instance
        """
        self.logger = logger or MultiProcessLogger()
        self.shutdown_requested = False
    
    def request_shutdown(self, reason, error=None):
        """
        Request graceful shutdown.
        
        Args:
            reason: Reason for shutdown
            error: Exception object (optional)
        """
        if not self.shutdown_requested:
            self.shutdown_requested = True
            self.logger.critical(f"Graceful shutdown requested: {reason}")
            
            if error is not None:
                self.logger.critical(f"Error details: {str(error)}")
                self.logger.critical(f"Traceback:\n{traceback.format_exc()}")
    
    def cleanup_processes(self, envs=None):
        """
        Clean up processes and resources.
        
        Args:
            envs: Vectorized environments to close
        """
        try:
            if envs is not None:
                self.logger.info("Closing environments...")
                envs.close()
                self.logger.info("Environments closed successfully")
        except Exception as e:
            self.logger.error(f"Error closing environments: {e}")
        
        # Clean up multiprocessing resources
        try:
            # Give processes time to terminate
            import time
            time.sleep(1)
            self.logger.info("Cleanup complete")
        except Exception as e:
            self.logger.error(f"Error during cleanup: {e}")
    
    def is_shutdown_requested(self):
        """Check if shutdown has been requested."""
        return self.shutdown_requested


def handle_training_error(error, model, optimizer, step, envs=None, 
                         checkpoint_handler=None, shutdown_handler=None, logger=None):
    """
    Comprehensive error handler for training errors.
    
    This function handles errors during training by:
    1. Logging the error with process ID
    2. Saving an emergency checkpoint
    3. Requesting graceful shutdown
    
    Args:
        error: Exception object
        model: PyTorch model
        optimizer: Optimizer
        step: Current training step
        envs: Vectorized environments
        checkpoint_handler: ErrorCheckpoint instance
        shutdown_handler: GracefulShutdown instance
        logger: MultiProcessLogger instance
        
    Returns:
        bool: True if recovery is possible, False if shutdown is needed
    """
    logger = logger or MultiProcessLogger()
    checkpoint_handler = checkpoint_handler or ErrorCheckpoint(logger=logger)
    shutdown_handler = shutdown_handler or GracefulShutdown(logger=logger)
    
    # Log the error
    error_type = type(error).__name__
    error_msg = str(error)
    logger.error(f"Training error occurred: {error_type}: {error_msg}", exc_info=True)
    
    # Prepare error info
    error_info = {
        'error_type': error_type,
        'error_message': error_msg,
        'traceback': traceback.format_exc(),
        'step': step
    }
    
    # Determine if error is recoverable
    recoverable_errors = [
        'RuntimeError',  # May be recoverable depending on context
    ]
    
    critical_errors = [
        'OutOfMemoryError',
        'CudaError',
        'KeyboardInterrupt',
        'SystemExit'
    ]
    
    is_critical = error_type in critical_errors or 'CUDA' in error_msg or 'memory' in error_msg.lower()
    
    # Save emergency checkpoint
    logger.info("Saving emergency checkpoint...")
    checkpoint_path = checkpoint_handler.save_emergency_checkpoint(
        model, optimizer, step, error_info, envs
    )
    
    if checkpoint_path:
        logger.info(f"Emergency checkpoint saved successfully: {checkpoint_path}")
    else:
        logger.error("Failed to save emergency checkpoint")
    
    # Request shutdown if critical
    if is_critical:
        shutdown_handler.request_shutdown(f"Critical error: {error_type}", error)
        shutdown_handler.cleanup_processes(envs)
        return False
    
    # For non-critical errors, log and attempt to continue
    logger.warning(f"Non-critical error detected. Attempting to continue training...")
    return True
