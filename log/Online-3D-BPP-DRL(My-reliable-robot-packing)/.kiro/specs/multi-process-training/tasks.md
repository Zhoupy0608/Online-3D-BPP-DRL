# Implementation Plan: Multi-Process Training

## Task List

- [x] 1. Verify and fix current multi-process environment setup





  - Review current ShmemVecEnv implementation in envs.py
  - Verify platform detection (Windows spawn vs Linux fork) is working
  - Test that environment parameters are properly serialized
  - Ensure unique seed assignment for each process (base_seed + rank)
  - _Requirements: 3.1, 3.2, 3.3, 3.5, 4.1_

- [x] 1.1 Write property test for unique seed assignment






  - **Property 2: Unique seeds across processes**
  - **Validates: Requirements 4.1**

- [x] 1.2 Write property test for environment parameter serialization






  - **Property 9: Environment parameter serialization**
  - **Validates: Requirements 3.3**

- [x] 2. Fix KFAC optimizer for multi-process training





  - Review KFAC implementation in acktr/algo/kfac.py
  - Verify Fisher matrix accumulation works with batched gradients
  - Implement device mismatch handling (move Q_a, d_a to Q_g device)
  - Verify eigenvalue thresholding (< 1e-6 → 0) is applied
  - Add gradient checking to detect NaN/Inf values
  - _Requirements: 2.1, 2.2, 2.3, 2.4, 2.5_

- [x] 2.1 Write property test for eigenvalue thresholding






  - **Property 6: Eigenvalue thresholding for stability**
  - **Validates: Requirements 2.3, 8.4**

- [x] 2.2 Write property test for device consistency






  - **Property 7: Device consistency in KFAC**
  - **Validates: Requirements 2.4**

- [x] 2.3 Write property test for no NaN in gradients






  - **Property 8: No NaN or Inf in gradients**
  - **Validates: Requirements 2.5**

- [x] 3. Update ACKTR agent for multi-process batching





  - Review acktr_pipeline.py update() method
  - Verify reshaping from (num_steps, num_processes, ...) to (num_steps * num_processes, ...)
  - Ensure Fisher loss computation includes all processes
  - Verify gradient accumulation across all processes
  - Test that batch size equals num_steps × num_processes
  - _Requirements: 1.3, 1.4, 2.1_

- [x] 3.1 Write property test for gradient batch size






  - **Property 3: Gradient batch size equals num_steps × num_processes**
  - **Validates: Requirements 1.4**

- [x] 3.2 Write property test for Fisher matrix accumulation






  - **Property 4: Fisher matrix accumulation across processes**
  - **Validates: Requirements 2.1**

- [x] 4. Verify rollout storage dimensions





  - Review storage.py RolloutStorage class
  - Verify all tensors have correct shape (num_steps, num_processes, ...)
  - Test insert() method with multi-process data
  - Test compute_returns() with episode boundaries
  - Verify after_update() correctly copies last step to first
  - _Requirements: 1.3, 4.4, 6.1_

- [x] 4.1 Write property test for rollout storage dimensions






  - **Property 1: Rollout storage dimensions match process count**
  - **Validates: Requirements 1.3, 6.1**

- [x] 4.2 Write property test for return computation






  - **Property 10: Return computation with episode boundaries**
  - **Validates: Requirements 4.4**

- [x] 5. Update training loop for multi-process statistics





  - Review main.py train_model() function
  - Implement episode reward aggregation from all processes
  - Add statistics computation (mean, median, min, max)
  - Update TensorBoard logging for multi-process metrics
  - Add reliability feature tracking across processes
  - _Requirements: 5.1, 5.2, 5.3, 5.4, 5.5_

- [x] 5.1 Write property test for episode reward aggregation






  - **Property 11: Episode reward aggregation**
  - **Validates: Requirements 5.1**

- [x] 5.2 Write property test for statistics computation






  - **Property 12: Statistics computation correctness**
  - **Validates: Requirements 5.2**

- [x] 5.3 Write property test for reliability feature tracking






  - **Property 13: Reliability feature tracking across processes**
  - **Validates: Requirements 5.5**

- [x] 6. Set paper-specified hyperparameters as defaults





  - Update arguments.py with paper defaults
  - Set num_processes default to 16
  - Set num_steps default to 5
  - Set loss coefficients: value_loss_coef=0.5, entropy_coef=0.01, invalid_coef=2
  - Set KFAC parameters: stat_decay=0.99, kl_clip=0.001, damping=1e-2
  - _Requirements: 7.1, 7.2, 7.3, 7.4_

- [x] 7. Implement error handling and logging





  - Add process ID to error messages
  - Implement device error detection and clear messages
  - Add gradient checking with NaN/Inf detection
  - Implement graceful shutdown on critical errors
  - Add checkpoint saving on errors
  - _Requirements: 8.1, 8.4, 8.5_

- [x] 8. Create multi-process training script





  - Create multi_process_train.py script
  - Set num_processes=16 by default
  - Add command-line options for process count
  - Include GPU detection and configuration
  - Add progress monitoring and ETA estimation
  - _Requirements: 1.1, 7.1_

- [ ] 9. Checkpoint - Verify multi-process training works




  - Ensure all tests pass, ask the user if questions arise
  - Run training with 4 processes for 100 updates
  - Verify no crashes or errors
  - Check that rewards improve over time
  - Compare with single-process training results

- [ ]* 10. Write integration tests for multi-process training
  - Test training loop with 4 processes for 100 updates
  - Test checkpoint saving and loading
  - Test reliability features with multi-process
  - Compare single-process vs multi-process consistency
  - _Requirements: 9.2, 9.3_

- [ ]* 11. Write performance tests
  - Test scaling with different process counts (1, 4, 8, 16)
  - Measure training speed and verify speedup
  - Monitor memory usage for leaks
  - Profile bottlenecks in the pipeline
  - _Requirements: 9.4_

- [ ] 12. Create documentation and usage guide
  - Document multi-process training setup
  - Add troubleshooting section for common issues
  - Document platform-specific considerations
  - Add performance tuning guide
  - Include expected results from paper
  - _Requirements: 1.1, 3.1, 3.2_

- [ ] 13. Final checkpoint - Validate against paper results
  - Ensure all tests pass, ask the user if questions arise
  - Run full training on CUT-2 dataset with 16 processes
  - Verify space utilization reaches ~70%
  - Compare training time with paper (~1 day on GPU)
  - Test on random sequence dataset
  - Verify stable learning without divergence
  - _Requirements: 9.1, 9.2_
