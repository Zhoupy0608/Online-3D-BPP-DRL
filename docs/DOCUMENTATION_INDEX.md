# Documentation Index

This document provides a comprehensive index of all documentation for the Online 3D Bin Packing with Reliability Features project.

## Main Documentation

### README.md
The main entry point for the project. Contains:
- Project overview and video links
- Installation instructions
- Basic usage examples
- Reliability features overview
- Training and testing commands
- Tips and best practices
- Citation information

## Quick Start & Troubleshooting

### START_HERE.md
The fastest way to get started:
- 2-step quick start guide
- Environment testing
- Training options comparison
- Common issues and solutions

### KFAC_ERROR_SOLUTION.md ⭐ NEW
Complete solution for K-FAC singular matrix errors:
- Immediate fix with `robust_train.py`
- Technical explanation of the error
- Improved K-FAC implementation details
- Alternative solutions (A2C algorithm)
- Training monitoring guide

### KFAC_TROUBLESHOOTING.md
Detailed troubleshooting guide for K-FAC optimizer:
- Problem diagnosis
- Multiple solution approaches
- Parameter tuning guide
- Technical deep dive
- Reference materials

### TRAINING_OPTIONS_COMPARISON.md
Side-by-side comparison of all training scripts:
- Feature comparison table
- Decision tree for choosing training method
- Performance benchmarks
- Error handling guide

## Reliability Features Documentation

### Quick Start
- **RELIABILITY_QUICK_REFERENCE.md**: Quick reference card with common commands, argument tables, and troubleshooting tips

### Comprehensive Guides
- **RELIABILITY_FEATURES_GUIDE.md**: Complete configuration guide covering:
  - Feature module descriptions
  - Configuration parameters
  - Usage scenarios
  - Performance tuning
  - Troubleshooting
  - API reference

### Specific Feature Guides
- **ENVIRONMENT_SELECTION_GUIDE.md**: How to choose between standard and reliable environments
- **UNIFIED_TEST_USAGE_GUIDE.md**: Comprehensive testing and evaluation guide
- **MULTI_CAMERA_USAGE_GUIDE.md**: Multi-camera setup and configuration

### Configuration Examples
- **camera_config_example.json**: Example camera calibration parameters for single and multi-camera setups

### Example Scripts
- **EXAMPLE_SCRIPTS_README.md**: Complete guide to example scripts
- **example_train_with_uncertainty.py**: Training script with uncertainty simulation
- **example_test_with_reliability.py**: Testing script with reliability features
- **example_full_pipeline.py**: Complete pipeline (train + test + compare)

## Specification Documents

Located in `.kiro/specs/reliable-robot-packing/`:

### requirements.md
Detailed requirements for reliability features including:
- User stories
- Acceptance criteria (EARS format)
- Glossary of terms
- 10 main requirements covering all features

### design.md
Technical design document including:
- Architecture diagrams
- Component interfaces
- Data models
- 20 correctness properties
- Error handling strategies
- Testing strategy

### tasks.md
Implementation task list with:
- 16 main tasks
- Sub-tasks for each feature
- Property-based test tasks
- Checkpoint tasks
- Task completion status

## Implementation Summaries

### TASK8_IMPLEMENTATION_SUMMARY.md
- Reward function enhancements
- Space utilization calculation
- Penalty mechanisms

### TASK9_IMPLEMENTATION_SUMMARY.md
- Training script updates
- Uncertainty simulation integration
- Logging enhancements

### TASK10_IMPLEMENTATION_SUMMARY.md
- Testing script updates
- Performance metrics
- Comparison functionality

### TASK11_IMPLEMENTATION_SUMMARY.md
- Multi-camera support
- Point cloud merging
- Coordinate transformation

### TASK8_VERIFICATION_CHECKLIST.md
- Verification steps for reward function
- Test coverage checklist

## Code Documentation

### Module Docstrings

All reliability feature modules have comprehensive docstrings:

#### acktr/uncertainty.py
- Module overview
- UncertaintySimulator class
- Methods: add_placement_noise(), validate_position()

#### acktr/motion_primitive.py
- Module overview
- MotionOption dataclass
- ParallelEntryMotion class
- Methods: generate_motion_options(), select_best_option(), check_collision()

#### acktr/visual_feedback.py
- Module overview
- Box and CameraConfig dataclasses
- VisualFeedbackModule class
- Methods: capture_point_cloud(), process_point_cloud(), update_height_map()
- Multi-camera methods: capture_and_merge_multi_camera(), merge_point_clouds()

#### envs/bpp0/bin3D_reliable.py
- Module overview
- ReliablePackingGame class
- Integration of all reliability features
- Enhanced step() method

### Test Documentation

All test files include:
- Module docstrings
- Test function docstrings
- Property-based test annotations

Test files:
- `acktr/test_uncertainty.py`
- `acktr/test_motion_primitive.py`
- `acktr/test_visual_feedback.py`
- `acktr/test_reliable_packing.py`
- `acktr/test_candidate_map.py`
- `acktr/test_reward_function.py`
- `acktr/test_performance_metrics.py`

## Command-Line Interface

### arguments.py
Complete argument definitions with help text for:
- Standard training/testing arguments
- Reliability feature flags
- Configuration parameters
- All arguments have descriptive help text

Run `python main.py --help` to see all available arguments.

## Usage Examples

### Quick Start with Example Scripts

**Complete Pipeline:**
```bash
python example_full_pipeline.py
```

**Train with Uncertainty:**
```bash
python example_train_with_uncertainty.py
```

**Test with Reliability Features:**
```bash
python example_test_with_reliability.py
```

See **EXAMPLE_SCRIPTS_README.md** for detailed documentation.

### Training Examples

**Standard Training:**
```bash
python main.py --mode train --use-cuda --item-seq rs
```

**Training with Uncertainty:**
```bash
python main.py --mode train --use-cuda --item-seq rs --uncertainty-enabled
```

**Training with Custom Noise:**
```bash
python main.py --mode train --use-cuda --item-seq rs \
    --uncertainty-enabled \
    --uncertainty-std-x 0.8 --uncertainty-std-y 0.8 --uncertainty-std-z 0.2
```

### Testing Examples

**Standard Testing:**
```bash
python main.py --mode test --load-model --use-cuda \
    --data-name cut_2.pt --load-name default_cut_2.pt
```

**Testing with Parallel Motion:**
```bash
python main.py --mode test --load-model --use-cuda \
    --data-name cut_2.pt --load-name default_cut_2.pt \
    --parallel-motion-enabled
```

**Testing with Visual Feedback:**
```bash
python main.py --mode test --load-model --use-cuda \
    --data-name cut_2.pt --load-name default_cut_2.pt \
    --visual-feedback-enabled --camera-config camera_config.json
```

**Testing with All Features:**
```bash
python main.py --mode test --load-model --use-cuda \
    --data-name cut_2.pt --load-name default_cut_2.pt \
    --visual-feedback-enabled --parallel-motion-enabled \
    --camera-config camera_config.json
```

### Evaluation Examples

**Comprehensive Evaluation:**
```bash
python unified_test.py --load-model --load-name default_cut_2.pt \
    --data-name cut_2.pt --cases 100 \
    --visual-feedback-enabled --parallel-motion-enabled
```

## Testing Documentation

### Running Tests

**All Tests:**
```bash
pytest
```

**Specific Module:**
```bash
pytest acktr/test_uncertainty.py
pytest acktr/test_motion_primitive.py
pytest acktr/test_visual_feedback.py
```

**With Verbose Output:**
```bash
pytest acktr/test_uncertainty.py -v
```

**Property-Based Tests Only:**
```bash
pytest -k "property" -v
```

## API Reference

### Python API

**Creating Reliable Environment:**
```python
from envs.bpp0.bin3D_reliable import ReliablePackingGame

env = ReliablePackingGame(
    uncertainty_enabled=True,
    visual_feedback_enabled=True,
    parallel_motion_enabled=True,
    noise_std=(0.5, 0.5, 0.1),
    buffer_range=(1, 1),
    camera_config=camera_config,
    container_size=(10, 10, 10),
    item_set=item_set,
    data_name='cut_2.pt'
)
```

**Using Uncertainty Simulator:**
```python
from acktr.uncertainty import UncertaintySimulator

simulator = UncertaintySimulator(
    noise_std=(0.5, 0.5, 0.1),
    enabled=True
)

perturbed_pos = simulator.add_placement_noise(
    position=(5, 5, 0),
    box_size=(2, 2, 3),
    height_map=height_map,
    container_size=(10, 10, 10)
)
```

**Using Parallel Entry Motion:**
```python
from acktr.motion_primitive import ParallelEntryMotion

motion = ParallelEntryMotion(
    buffer_range=(1, 1),
    container_size=(10, 10, 10)
)

options = motion.generate_motion_options(
    target_pos=(5, 5, 0),
    box_size=(2, 2, 3),
    height_map=height_map
)

best_option = motion.select_best_option(options)
```

**Using Visual Feedback:**
```python
from acktr.visual_feedback import VisualFeedbackModule

visual_feedback = VisualFeedbackModule(
    camera_config=camera_config,
    container_size=(10, 10, 10),
    grid_size=(10, 10),
    simulation_mode=False
)

point_cloud = visual_feedback.capture_point_cloud()
boxes = visual_feedback.process_point_cloud(point_cloud)
updated_map = visual_feedback.update_height_map(height_map, boxes)
```

## Troubleshooting

For troubleshooting information, see:
- **RELIABILITY_FEATURES_GUIDE.md**: Comprehensive troubleshooting section
- **RELIABILITY_QUICK_REFERENCE.md**: Quick troubleshooting table
- Test files for usage examples

## Additional Resources

### Papers
- Original paper: "Online 3D Bin Packing with Constrained Deep Reinforcement Learning" (AAAI 2021)
- Reliability paper: "Towards reliable robot packing system based on deep reinforcement learning" (Advanced Engineering Informatics 2023)

### External Documentation
- Open3D documentation: http://www.open3d.org/docs/
- Hypothesis documentation: https://hypothesis.readthedocs.io/
- PyTorch documentation: https://pytorch.org/docs/

## Document Organization

```
.
├── README.md                              # Main documentation
├── RELIABILITY_FEATURES_GUIDE.md          # Comprehensive guide
├── RELIABILITY_QUICK_REFERENCE.md         # Quick reference
├── DOCUMENTATION_INDEX.md                 # This file
├── camera_config_example.json             # Camera config template
├── ENVIRONMENT_SELECTION_GUIDE.md         # Environment selection
├── UNIFIED_TEST_USAGE_GUIDE.md           # Testing guide
├── MULTI_CAMERA_USAGE_GUIDE.md           # Multi-camera guide
├── TASK*_IMPLEMENTATION_SUMMARY.md        # Implementation summaries
├── .kiro/specs/reliable-robot-packing/
│   ├── requirements.md                    # Requirements spec
│   ├── design.md                          # Design spec
│   └── tasks.md                           # Task list
├── acktr/
│   ├── arguments.py                       # CLI arguments
│   ├── uncertainty.py                     # Uncertainty module
│   ├── motion_primitive.py                # Motion module
│   ├── visual_feedback.py                 # Visual feedback module
│   ├── test_*.py                          # Test files
│   └── ...
└── envs/bpp0/
    └── bin3D_reliable.py                  # Reliable environment
```

## Getting Help

1. **Quick answers**: Check RELIABILITY_QUICK_REFERENCE.md
2. **Configuration**: Read RELIABILITY_FEATURES_GUIDE.md
3. **Troubleshooting**: See troubleshooting sections in guides
4. **Examples**: Look at test files in acktr/test_*.py
5. **Design details**: Read .kiro/specs/reliable-robot-packing/design.md
6. **API usage**: Check module docstrings in source files

## Contributing

When adding new features or documentation:
1. Update relevant specification documents
2. Add docstrings to all classes and methods
3. Create or update test files
4. Update this index if adding new documentation files
5. Update README.md with usage examples
6. Add entries to RELIABILITY_QUICK_REFERENCE.md if applicable
