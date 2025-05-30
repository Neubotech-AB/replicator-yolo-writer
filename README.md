# YOLO Writer for NVIDIA Omniverse Replicator

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.7+](https://img.shields.io/badge/python-3.7+-blue.svg)](https://www.python.org/downloads/)
[![NVIDIA Omniverse](https://img.shields.io/badge/NVIDIA-Omniverse-76B900)](https://www.nvidia.com/en-us/omniverse/)

A custom writer for NVIDIA Omniverse Replicator that converts synthetic training data from Omniverse and Isaac Sim into YOLO format for computer vision tasks.

## Overview

This module provides a robust solution for generating YOLO datasets from synthetic data created with NVIDIA Omniverse Replicator, including scenarios built in Isaac Sim. It seamlessly converts Replicator's annotations into YOLO formats for:

- **Object Detection**: Normalized bounding boxes in YOLO format
- **Instance Segmentation**: Polygon-based masks in YOLO format

The writer features automatic train/validation splits and generates YAML configuration files for immediate use with YOLO training pipelines.

## Key Features

- 🎯 **YOLO Object Detection**: Normalized bounding boxes in standard YOLO format (`<class_id> <x_center> <y_center> <width> <height>`)
- 🔍 **YOLO Instance Segmentation**: Normalized polygon coordinates (`<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>`)
- 📊 **Data Splitting**: Configurable train/validation splits with organized directory structure
- ⚙️ **Auto-Configuration**: Generates YOLO-compatible dataset YAML files
- 🏷️ **Flexible Class Mapping**: Custom class name to ID mapping with validation
- 🖼️ **Multi-Format Support**: JPG, PNG, and other image format compatibility
- 🚀 **Isaac Sim Ready**: Optimized for synthetic data from Isaac Sim simulation scenarios

## Installation

### Prerequisites

- **NVIDIA Omniverse** with Replicator extension
- **Isaac Sim** (optional, for robotics scenarios)
- **Python 3.7+**
- **Required Python packages**: `opencv-python`, `numpy`

### Quick Start

1. **Clone or download** this repository to your project directory
2. **Install dependencies**:
   ```bash
   pip install opencv-python numpy
   ```
3. **Import and use** in your Replicator or Isaac Sim project:
   ```python
   from yolo_writer import YOLOWriter
   ```

> **Note**: Ensure NVIDIA Omniverse with Replicator is properly installed and configured before use.

## Usage Examples

### Basic Implementation

```python
import omni.replicator.core as rep
from yolo_writer import YOLOWriter

# Register the writer
rep.WriterRegistry.register(YOLOWriter)

# Define your class mapping (customize for your use case)
class_mapping = {
    "person": 0,
    "vehicle": 1,
    "robot": 2,
    "obstacle": 3
}

# Create writer instance
writer = rep.WriterRegistry.get("YOLOWriter")

# Initialize the YOLO writer
writer = YOLOWriter(
    output_dir="/path/to/your/dataset",
    rgb=True,
    bounding_box_2d_tight=True,   # Enable object detection
    instance_segmentation=True,   # Enable instance segmentation
    class_mapping=class_mapping,
    train_val_split=0.7,          # 70% train, 30% validation
    image_output_format="jpg",
    min_bbox_area=0.001,          # Minimum bounding box area
    min_mask_area=0.001,          # Minimum mask area
    max_points=100                # Maximum vertices for polygon
)
```

## Dataset Structure

The writer creates a professionally organized dataset structure compatible with all major YOLO implementations:

```
📁 output_dir/
├── 📁 detection/                  # Object detection dataset (YOLO format)
│   ├── 📁 images/
│   │   ├── 📁 train/              # Training images
│   │   └── 📁 val/                # Validation images
│   ├── 📁 labels/
│   │   ├── 📁 train/              # Training annotations (.txt files)
│   │   └── 📁 val/                # Validation annotations (.txt files)
│   └── 📄 dataset.yaml            # YOLO dataset configuration
│
├── 📁 segmentation/               # Instance segmentation dataset (YOLO format)
│   ├── 📁 images/
│   │   ├── 📁 train/
│   │   └── 📁 val/
│   ├── 📁 labels/
│   │   ├── 📁 train/
│   │   └── 📁 val/
│   └── 📄 dataset.yaml
│
└── 📄 metadata.json               # Writer metadata and statistics
```

## Configuration Reference

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `output_dir` | `str` | **Required** | Output directory for the generated dataset |
| `rgb` | `bool` | `True` | Enable RGB image output |
| `bounding_box_2d_tight` | `bool` | `False` | Enable object detection dataset generation |
| `instance_segmentation` | `bool` | `False` | Enable instance segmentation dataset generation |
| `class_mapping` | `dict` | **Required** | Dictionary mapping class names to integer IDs |
| `min_bbox_area` | `float` | `0.001` | Minimum normalized bounding box area (0.1% of image) |
| `min_mask_area` | `float` | `0.001` | Minimum normalized mask area (0.1% of image) |
| `max_points` | `int` | `100` | Maximum vertices for polygon approximation |
| `image_output_format` | `str` | `"jpg"` | Image format: `"jpg"`, `"png"`, etc. |
| `train_val_split` | `float` | `0.7` | Training split ratio (range: 0.1-0.9) |

## Data Format Specifications

### Object Detection Format
Each annotation file contains one line per detected object:
```
<class_id> <x_center> <y_center> <width> <height>
```
- All coordinates are normalized to the range `[0, 1]`
- `x_center`, `y_center`: Normalized center coordinates
- `width`, `height`: Normalized dimensions

**Example:**
```
0 0.5 0.3 0.2 0.4
1 0.7 0.8 0.1 0.15
```

### Instance Segmentation Format
Each annotation file contains one line per segmented object:
```
<class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
```
- All coordinates are normalized to the range `[0, 1]`
- Points represent polygon vertices in order
- Automatic polygon simplification maintains shape fidelity

**Example:**
```
0 0.1 0.2 0.3 0.1 0.4 0.3 0.2 0.4
1 0.6 0.7 0.8 0.6 0.9 0.8 0.7 0.9
```

### YAML Configuration Format
Generated `dataset.yaml` files follow the standard YOLO specification:
```yaml
# Dataset paths (relative to yaml file)
train: images/train
val: images/val

# Number of classes
nc: 4

# Class names mapping
names:
  0: robot_arm
  1: gripper
  2: target_object
  3: obstacle
```

## Integration with YOLO Training

### YOLOv8 Integration
The generated datasets are immediately compatible with Ultralytics YOLOv8:

```python
from ultralytics import YOLO

# Object Detection Training
detection_model = YOLO('yolov8n.pt')  # or yolov8s.pt, yolov8m.pt, etc.
detection_model.train(
    data='/path/to/dataset/detection/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)

# Instance Segmentation Training
segmentation_model = YOLO('yolov8n-seg.pt')  # segmentation variant
segmentation_model.train(
    data='/path/to/dataset/segmentation/dataset.yaml',
    epochs=100,
    imgsz=640,
    batch=16
)
```

## Contributing

We welcome contributions from the community! Please:

1. **Fork** the repository
2. **Create** a feature branch (`git checkout -b feature/amazing-feature`)
3. **Commit** your changes (`git commit -m 'Add amazing feature'`)
4. **Push** to the branch (`git push origin feature/amazing-feature`)
5. **Open** a Pull Request

### Reporting Issues
Please use the GitHub Issues tab to report bugs or request features. Include:
- Your Omniverse/Isaac Sim version
- Python version and dependencies
- Minimal reproduction example
- Expected vs. actual behavior

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for complete details.
