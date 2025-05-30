#!/usr/bin/env python3
"""
YOLO Format Writer for NVIDIA Omniverse Replicator.

This module implements a custom writer for Replicator that outputs annotations in
YOLO format for object detection and instance segmentation.

For object detection:
- Format: <class_id> <x_center> <y_center> <width> <height>
  All values are normalized to [0, 1]

For instance segmentation:
- Format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
  All coordinates are normalized to [0, 1]
"""

import os
import json
import random
import logging
import numpy as np
import cv2
from datetime import datetime
from omni.replicator.core import Writer, AnnotatorRegistry, BackendDispatch


class YOLOWriter(Writer):
    """
    Custom writer for outputting annotations in YOLO format.
    
    This writer supports both object detection (bounding boxes) and 
    instance segmentation masks in the YOLO format.
    """
    
    # Define version for writer metadata
    VERSION = "1.0.0"
    
    def __init__(
        self,
        output_dir,
        rgb=True,
        bounding_box_2d_tight=False,
        instance_segmentation=False,
        class_mapping=None,         # Mapping from class names to class indices
        min_bbox_area=0.001,          # Minimum normalized bbox area to consider (0.001 = 0.1%)
        min_mask_area=0.001,         # Minimum normalized mask area to consider (0.001 = 0.1%)
        max_points=100,             # Maximum number of points for polygon approximation
        image_output_format="jpg",  # Image output format (jpg, png, etc.)
        train_val_split=0.7,        # Split ratio (0.7 = 70% train, 30% val)
    ):
        """Initialize the YOLO format writer."""
        # Base configuration
        self._output_dir = output_dir
        self._backend = BackendDispatch({"paths": {"out_dir": output_dir}})
        self._frame_id = -1 
        self._sequence_id = ""
        self._image_output_format = image_output_format
        self._min_bbox_area = min_bbox_area
        self._min_mask_area = min_mask_area
        self._max_points = max_points
        
        # Class mapping
        self._class_mapping = class_mapping
        if self._class_mapping is None:
            raise ValueError("Class mapping must be provided for YOLOWriter.")

        # Train/val split configuration
        self._train_val_split = max(0.1, min(0.9, train_val_split))
        
        # Setup metadata
        self.metadata = {
            "version": self.VERSION,
            "backend": "file",
            "format": "YOLO",
            "train_val_split": self._train_val_split,
            "date_created": datetime.now().isoformat()
        }
        
        # Create output directories
        self._output_detection = bounding_box_2d_tight
        self._output_segmentation = instance_segmentation
        
        # Create directories for detection and segmentation
        self._create_directory_structure(bounding_box_2d_tight, instance_segmentation)
        
        # Setup annotators
        self.annotators = []
        if rgb:
            self.annotators.append(AnnotatorRegistry.get_annotator("rgb"))
        
        if bounding_box_2d_tight:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "bounding_box_2d_tight",
                    init_params={"semanticTypes": ["class"]}
                )
            )
        
        if instance_segmentation:
            self.annotators.append(
                AnnotatorRegistry.get_annotator(
                    "instance_segmentation",
                    init_params={"semanticTypes": ["class"]}
                )
            )

        # Set up logging
        logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger("YOLOWriter")

    @property
    def version(self):
        """Return the version of this writer."""
        return self.VERSION
    
    def get_metadata(self):
        """Return metadata for this writer."""
        return self.metadata
    
    def write(self, data):
        """Write data to YOLO format files."""
        # Increment frame ID for each write call
        self._frame_id += 1
        # Process sequence ID from trigger outputs
        sequence_id = self._process_sequence_id(data)
                
        # Check if we have RGB data, which is required
        if "rgb" not in data:
            return False
            
        # Determine if this sample should go to train or val set
        is_val_sample = random.random() > self._train_val_split

        # Generate file paths for images and labels
        paths = self._generate_file_paths(is_val_sample)
        
        # Get image dimensions
        rgb_data = data["rgb"]
        img_height, img_width = self._get_image_dimensions(rgb_data)
        
        # Process bounding boxes and instance segmentation
        if self._output_detection:
            det_annotations = self._process_bounding_boxes(data, img_width, img_height)
        
        if self._output_segmentation:
            seg_annotations = self._process_instance_segmentation(data, img_width, img_height)

        # Write images and annotations to files
        self._write_output_files(rgb_data, paths, det_annotations, seg_annotations)
        
        # Update metadata and class mapping
        self._update_metadata_periodically()
        
        # Return success
        return True
    
    def on_final_frame(self):
        """
        Called when all frames have been processed.
        
        Ensures all data is written and finalizes the dataset.
        """
        self.logger.info("Processing final frame, finalizing dataset...")
        self._write_all_dataset_yaml()
        self.logger.info(f"Dataset finalized with {self._frame_id} frames and {len(self._class_mapping)} classes")
        
        # Log train/val split info
        self.logger.info(f"Data split: {self._train_val_split*100:.1f}% train, {(1-self._train_val_split)*100:.1f}% validation")

    def _process_sequence_id(self, data):
        """Process the sequence ID from trigger outputs."""
        sequence_id = ""
        if "trigger_outputs" in data:
            for trigger_name, call_count in data["trigger_outputs"].items():
                if "on_time" in trigger_name:
                    sequence_id = f"{call_count}_{sequence_id}"
            
            if sequence_id != self._sequence_id:
                self._frame_id = 0
                self._sequence_id = sequence_id
        
        return sequence_id    

    def _create_directory_structure(self, bounding_box_2d_tight, instance_segmentation):
        """Create the necessary directory structure for YOLO dataset."""
        # Setup detection directories
        if bounding_box_2d_tight:
            self._det_dir = os.path.join(self._output_dir, "detection")
            os.makedirs(self._det_dir, exist_ok=True)
            
            # Create detection images directories
            self._det_img_train_dir = os.path.join(self._det_dir, "images", "train")
            self._det_img_val_dir = os.path.join(self._det_dir, "images", "val")
            os.makedirs(self._det_img_train_dir, exist_ok=True)
            os.makedirs(self._det_img_val_dir, exist_ok=True)
            
            # Create detection labels directories
            self._det_label_train_dir = os.path.join(self._det_dir, "labels", "train")
            self._det_label_val_dir = os.path.join(self._det_dir, "labels", "val")
            os.makedirs(self._det_label_train_dir, exist_ok=True)
            os.makedirs(self._det_label_val_dir, exist_ok=True)
        else:
            self._det_dir = None
        
        # Setup segmentation directories
        if instance_segmentation:
            self._seg_dir = os.path.join(self._output_dir, "segmentation")
            os.makedirs(self._seg_dir, exist_ok=True)
            
            # Create segmentation images directories
            self._seg_img_train_dir = os.path.join(self._seg_dir, "images", "train")
            self._seg_img_val_dir = os.path.join(self._seg_dir, "images", "val")
            os.makedirs(self._seg_img_train_dir, exist_ok=True)
            os.makedirs(self._seg_img_val_dir, exist_ok=True)
            
            # Create segmentation labels directories
            self._seg_label_train_dir = os.path.join(self._seg_dir, "labels", "train")
            self._seg_label_val_dir = os.path.join(self._seg_dir, "labels", "val")
            os.makedirs(self._seg_label_train_dir, exist_ok=True)
            os.makedirs(self._seg_label_val_dir, exist_ok=True)
        else:
            self._seg_dir = None

    def _generate_file_paths(self, is_val_sample):
        """Generate file paths for images and labels."""
        image_filename = f"frame_{self._sequence_id}{self._frame_id:06d}.{self._image_output_format}"
        label_filename = f"frame_{self._sequence_id}{self._frame_id:06d}.txt"
        
        paths = {
            "det_img": None,
            "det_label": None,
            "seg_img": None,
            "seg_label": None
        }
        
        # Setup detection paths
        if self._output_detection:
            if is_val_sample:
                paths["det_img"] = os.path.join(self._det_img_val_dir, image_filename)
                paths["det_label"] = os.path.join(self._det_label_val_dir, label_filename)
            else:
                paths["det_img"] = os.path.join(self._det_img_train_dir, image_filename)
                paths["det_label"] = os.path.join(self._det_label_train_dir, label_filename)
        
        # Setup segmentation paths
        if self._output_segmentation:
            if is_val_sample:
                paths["seg_img"] = os.path.join(self._seg_img_val_dir, image_filename)
                paths["seg_label"] = os.path.join(self._seg_label_val_dir, label_filename)
            else:
                paths["seg_img"] = os.path.join(self._seg_img_train_dir, image_filename)
                paths["seg_label"] = os.path.join(self._seg_label_train_dir, label_filename)
        
        return paths

    def _write_metadata(self):
        """Write metadata to a JSON file in the proper format."""
        try:
            # Update metadata with current information
            self.metadata["date_updated"] = datetime.now().isoformat()
            self.metadata["class_mapping"] = self._class_mapping
            self.metadata["num_frames"] = self._frame_id + 1
            
            # Write metadata file
            with open(os.path.join(self._output_dir, "metadata.json"), "w") as f:
                json.dump(self.metadata, f, indent=2)
                
            return True
        except Exception as e:
            self.logger.error(f"Error writing metadata: {e}")
            return False

    def _write_all_dataset_yaml(self):
        """Write all dataset YAML configuration files for detection and segmentation."""
        try:
            # Write detection dataset configuration
            if self._output_detection:
                self._write_dataset_yaml(self._det_dir, "detection")

            # Write segmentation dataset configuration
            if self._output_segmentation:
                self._write_dataset_yaml(self._seg_dir, "segmentation")

            # Also write metadata
            self._write_metadata()
            return True
        except Exception as e:
            self.logger.error(f"Error writing dataset YAML config: {e}")
            return False
    
    def _write_dataset_yaml(self, directory, dataset_type):
        """Helper method to write a single dataset.yaml file."""
        try:
            yaml_content = [f"# YOLO dataset configuration for {dataset_type}\n"]
            yaml_content.append(f"path: .\n")
            
            # Use train/val folders
            yaml_content.append("train: images/train\n")
            yaml_content.append("val: images/val\n")
            
            # Get and add class names
            yaml_class_names = self._generate_class_mapping_yaml()
            yaml_content.extend(yaml_class_names)
            
            # Write YAML file
            yaml_path = os.path.join(directory, "dataset.yaml")
            with open(yaml_path, "w") as f:
                f.writelines(yaml_content)
            self.logger.info(f"Wrote {dataset_type} dataset YAML to {yaml_path}")
            return True
        except Exception as e:
            self.logger.error(f"Error writing {dataset_type} dataset YAML: {e}")
            return False

    def _write_output_files(self, rgb_data, paths, det_annotations, seg_annotations):
        """Write image and annotation files."""
        # Write detection files
        if self._output_detection and paths["det_img"] and paths["det_label"]:
            # Write the image
            self._backend.write_image(paths["det_img"], rgb_data)
            
            # Write detection annotations
            with open(paths["det_label"], "w") as f:
                f.write("\n".join(det_annotations))
        
        # Write segmentation files
        if self._output_segmentation and paths["seg_img"] and paths["seg_label"]:
            # Don't write the image again if it's the same path (can happen if detection=False)
            if not self._output_detection or paths["seg_img"] != paths["det_img"]:
                self._backend.write_image(paths["seg_img"], rgb_data)
            
            # Write segmentation annotations
            with open(paths["seg_label"], "w") as f:
                f.write("\n".join(seg_annotations))

    def _generate_class_mapping_yaml(self):
        """Generate YAML content for class names.
        
        Returns:
            list: A list of strings representing class names YAML content.
        """
        yaml_class_names = ["# Class names\n", "names:\n"]
        
        # Sort by class ID for consistent output
        sorted_classes = sorted(self._class_mapping.items(), key=lambda x: x[1])
        for class_name, class_id in sorted_classes:
            yaml_class_names.append(f"  {class_id}: {class_name}\n")
        
        return yaml_class_names
    
    def _update_metadata_periodically(self):
        """Update metadata and class mapping periodically."""
        # Write dataset YAML every 100 frames
        if self._frame_id % 100 == 0 or self._frame_id == 0:
            self._write_all_dataset_yaml()
        else:
            # Always write metadata for each frame to ensure it's available
            self._write_metadata()

    def _get_image_dimensions(self, img_data):
        """Get image dimensions from RGB data."""
        if not isinstance(img_data, np.ndarray):
            # Convert to numpy array for dimension extraction
            img_data = np.array(img_data)
        img_height, img_width = img_data.shape[:2]
        return img_height, img_width

    def _check_bbox_area(self, box_data, img_width, img_height):
        """Check if the bounding box has sufficient area."""
        width = abs(box_data["x_max"] - box_data["x_min"])
        height = abs(box_data["y_max"] - box_data["y_min"])
        box_area = width * height / (float(img_width * img_height))
        return box_area > self._min_bbox_area

    def _check_mask_area(self, mask_data, img_width, img_height):
        """Check if the mask has sufficient area."""
        mask_area = np.sum(mask_data > 0) / (float(img_height * img_width))
        return mask_area > self._min_mask_area
    
    def _process_bounding_boxes(self, data, img_width, img_height):
        """Process bounding box data and extract annotations."""
        annotations = []
        try:
            # Extract bounding box data and labels
            bbox_data = data["bounding_box_2d_tight"]["data"]
            id_to_labels = data["bounding_box_2d_tight"]["info"]["idToLabels"]
            
            # Process each semantic class
            for semantic_id, labels in id_to_labels.items():
                semantic_id = int(semantic_id)
                class_id = self._class_mapping.get(labels["class"], None)
                if class_id is None:
                    continue
                try:
                    # Extract bounding boxes for this semantic ID
                    bbox_array = bbox_data[bbox_data["semanticId"] == semantic_id]
                    # Convert the bounding boxes to YOLO format
                    yolo_bbox_array = self._convert_to_yolo_bbox(bbox_array, img_width, img_height)
                    # Add bounding box annotation
                    for yolo_bbox in yolo_bbox_array:
                        # Format: <class_id> <x_center> <y_center> <width> <height>
                        bbox_annotation = f"{class_id} {yolo_bbox[0]:.6f} {yolo_bbox[1]:.6f} {yolo_bbox[2]:.6f} {yolo_bbox[3]:.6f}"
                        annotations.append(bbox_annotation)
                except Exception as e:
                    self.logger.error(f"Error processing bounding box for instance {semantic_id}: {e}")
                    continue
            return annotations
        except Exception as e:
            self.logger.error(f"Error processing bounding boxes: {e}")

    def _convert_to_yolo_bbox(self, bbox_array, img_width, img_height):
        """Convert bounding boxes from Replicator format to YOLO format."""
        bbox = []
        for bbox_data in bbox_array:
            # Convert to normalized center x, center y, width, height format
            x_center = (bbox_data["x_min"] + bbox_data["x_max"]) / 2.0 / img_width
            y_center = (bbox_data["y_min"] + bbox_data["y_max"]) / 2.0 / img_height
            width = abs(bbox_data["x_max"] - bbox_data["x_min"]) / img_width
            height = abs(bbox_data["y_max"] - bbox_data["y_min"]) / img_height

            # Clamp values to be within [0, 1]
            x_center = max(0.0, min(1.0, x_center))
            y_center = max(0.0, min(1.0, y_center))
            width = max(0.0, min(1.0, width))
            height = max(0.0, min(1.0, height))

            # Check if the bounding box area is above the minimum threshold
            if self._check_bbox_area(bbox_data, img_width, img_height):
                bbox.append((x_center, y_center, width, height))

        return np.array(bbox, dtype=np.float32)
    
    def _process_instance_segmentation(self, data, img_width, img_height):
        """Process instance segmentation data and extract annotations."""
        annotations = []
        try:
            # Extract mask data and labels
            mask_data = data["instance_segmentation"]["data"]
            id_to_semantics = data["instance_segmentation"]["info"]["idToSemantics"]

            # Process each semantic class
            for semantic_id, labels in id_to_semantics.items():
                semantic_id = int(semantic_id)
                labels = labels["class"].split(",")
                if len(labels) == 1:
                    class_id = self._class_mapping.get(labels[0], None)
                else:
                    for label in labels:
                        class_id = self._class_mapping.get(label, None)
                        if class_id is not None:
                            break
                # If we couldn't find a class ID, skip
                if class_id is None:
                    continue
                try:                        
                    # Extract polygon points
                    polygon_points = self._extract_polygon_points(mask_data, semantic_id, img_width, img_height)
                    # If valid polygon points are found, format them
                    if polygon_points:
                        # Format: <class_id> <x1> <y1> <x2> <y2> ... <xn> <yn>
                        points_str = " ".join([f"{p:.6f}" for p in polygon_points])
                        mask_annotation = f"{class_id} {points_str}"
                        annotations.append(mask_annotation)
                except Exception as e:
                    self.logger.error(f"Error processing instance segmentation for instance {semantic_id}: {e}")
                    continue
            return annotations
        except Exception as e:
            self.logger.error(f"Error processing instance segmentation: {e}")

    def _extract_polygon_points(self, mask_data, semantic_id, img_width, img_height):
        """Extract polygon points from instance segmentation mask."""
        try:
            # Extract mask for the given semantic ID
            mask = (mask_data == semantic_id).astype(np.uint8) * 255
            # Check if the mask area is above the minimum threshold
            if not self._check_mask_area(mask, img_width, img_height):
                return []
                
            # Convert binary mask to polygon using OpenCV
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            if not contours:
                return []
            
            # Get the largest contour
            largest_contour = max(contours, key=cv2.contourArea)
                
            # Approximate the polygon to reduce the number of points
            epsilon = 0.005 * cv2.arcLength(largest_contour, True)
            approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Limit the number of points if necessary
            if len(approx_polygon) > self._max_points:
                # Further approximate to reduce points
                epsilon_scale = 1.5
                while len(approx_polygon) > self._max_points:
                    epsilon *= epsilon_scale
                    approx_polygon = cv2.approxPolyDP(largest_contour, epsilon, True)
            
            # Convert to normalized coordinates
            normalized_points = []
            for point in approx_polygon:
                x, y = point[0]
                norm_x = max(0.0, min(1.0, x / img_width))
                norm_y = max(0.0, min(1.0, y / img_height))
                normalized_points.extend([norm_x, norm_y])
            
            return normalized_points
            
        except Exception as e:
            self.logger.error(f"Error extracting polygon points: {e}")
            return []