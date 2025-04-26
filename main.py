import cv2
import numpy as np
import time
import argparse
import os
import json
from ultralytics import YOLO
from collections import defaultdict
import supervision as sv
from typing import Dict, List, Tuple, Optional


class ObjectLogger:
    def __init__(self, log_dir="object_logs"):
        self.log_dir = log_dir
        # Create log directory if it doesn't exist
        os.makedirs(log_dir, exist_ok=True)
        self.missing_objects_file = os.path.join(log_dir, "missing_objects.json")
        self.new_objects_file = os.path.join(log_dir, "new_objects.json")
        
        # Initialize files if they don't exist
        for file_path in [self.missing_objects_file, self.new_objects_file]:
            if not os.path.exists(file_path):
                with open(file_path, 'w') as f:
                    json.dump([], f)
    
    def _load_json_file(self, file_path):
        try:
            with open(file_path, 'r') as f:
                return json.load(f)
        except (json.JSONDecodeError, FileNotFoundError):
            return []
    
    def _save_json_file(self, file_path, data):
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _convert_to_serializable(self, obj):
        """Convert any non-serializable types to standard Python types."""
        if isinstance(obj, dict):
            return {key: self._convert_to_serializable(value) for key, value in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [self._convert_to_serializable(item) for item in obj]
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj
    
    def log_missing_object(self, object_data):
        """Log when an object disappears from the frame"""
        objects = self._load_json_file(self.missing_objects_file)
        
        # Convert object_data to serializable format
        object_data = self._convert_to_serializable(object_data)
        
        # Add timestamp and format for better readability
        object_data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        object_data['disappeared_time'] = time.strftime("%Y-%m-%d %H:%M:%S", 
                                                       time.localtime(object_data['disappeared_time']))
        
        objects.append(object_data)
        self._save_json_file(self.missing_objects_file, objects)
        
    def log_new_object(self, object_data):
        """Log when a new object appears in the frame"""
        objects = self._load_json_file(self.new_objects_file)
        
        # Convert object_data to serializable format
        object_data = self._convert_to_serializable(object_data)
        
        # Add timestamp and format for better readability
        object_data['timestamp'] = time.strftime("%Y-%m-%d %H:%M:%S")
        object_data['appeared_time'] = time.strftime("%Y-%m-%d %H:%M:%S", 
                                                    time.localtime(object_data['appeared_time']))
        
        objects.append(object_data)
        self._save_json_file(self.new_objects_file, objects)


class SceneMemory:
    def __init__(self, memory_duration=5, confidence_threshold=0.5, iou_threshold=0.3, logger=None):
        self.memory_duration = memory_duration
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.object_history = {}
        self.missing_objects = []
        self.new_objects = []
        self.last_frame_time = time.time()
        self.logger = logger
        # Keep track of already logged objects to avoid duplicates
        self.logged_missing_ids = set()
        self.logged_new_ids = set()

    def update(self, detections, frame):
        current_time = time.time()
        time_diff = current_time - self.last_frame_time
        self.last_frame_time = current_time

        current_track_ids = set()
        if detections is not None and len(detections) > 0:
            current_track_ids = set(detections.tracker_id.tolist())

        # Check for missing objects
        for track_id, data in list(self.object_history.items()):
            if track_id not in current_track_ids:
                data['frames_missing'] += 1
                if data['frames_missing'] >= 10:  # Object is considered missing after 10 frames
                    missing_obj = {
                        'track_id': int(track_id),  # Convert numpy int to Python int
                        'class_id': int(data['class_id']),  # Convert numpy int to Python int
                        'class_name': data['class_name'],
                        'last_bbox': data['bbox'],
                        'disappeared_time': current_time
                    }
                    self.missing_objects.append(missing_obj)
                    
                    # Log the missing object if we have a logger and haven't logged it yet
                    if self.logger and track_id not in self.logged_missing_ids:
                        self.logger.log_missing_object(missing_obj.copy())
                        self.logged_missing_ids.add(track_id)
                    
                    del self.object_history[track_id]
            else:
                data['frames_missing'] = 0

        # Process detections for current frame
        if detections is not None and len(detections) > 0:
            for i in range(len(detections.tracker_id)):
                track_id = detections.tracker_id[i]
                class_id = detections.class_id[i]
                confidence = detections.confidence[i]
                bbox = detections.xyxy[i]
                if confidence < self.confidence_threshold:
                    continue
                class_name = detections.class_names[class_id]
                
                if track_id in self.object_history:
                    # Update existing object
                    self.object_history[track_id].update({
                        'last_seen': current_time,
                        'class_id': class_id,
                        'class_name': class_name,
                        'bbox': bbox,
                        'confidence': confidence,
                        'consecutive_frames': self.object_history[track_id]['consecutive_frames'] + 1
                    })
                else:
                    # New object detected
                    self.object_history[track_id] = {
                        'first_seen': current_time,
                        'last_seen': current_time,
                        'class_id': class_id,
                        'class_name': class_name,
                        'bbox': bbox,
                        'confidence': confidence,
                        'frames_missing': 0,
                        'consecutive_frames': 1
                    }
                
                # Check if object is consistently present enough to be considered "new"
                if track_id in self.object_history and self.object_history[track_id]['consecutive_frames'] >= 3:
                    # Only add to new_objects if we haven't already
                    if track_id not in self.logged_new_ids:
                        new_obj = {
                            'track_id': int(track_id),  # Convert numpy int to Python int
                            'class_id': int(class_id),  # Convert numpy int to Python int
                            'class_name': class_name,
                            'bbox': bbox,
                            'appeared_time': current_time
                        }
                        self.new_objects.append(new_obj)
                        
                        # Log the new object if we have a logger
                        if self.logger:
                            self.logger.log_new_object(new_obj.copy())
                            self.logged_new_ids.add(track_id)

        # Clean up objects that have been missing or new for too long
        self.missing_objects = [obj for obj in self.missing_objects 
                                if current_time - obj['disappeared_time'] <= self.memory_duration]
        self.new_objects = [obj for obj in self.new_objects 
                            if current_time - obj['appeared_time'] <= self.memory_duration]

        return self.missing_objects, self.new_objects

    def get_status(self):
        return {
            'tracked_objects': len(self.object_history),
            'missing_objects': len(self.missing_objects),
            'new_objects': len(self.new_objects)
        }


def detections_from_yolov8(result):
    if hasattr(result, 'boxes'):
        boxes = result.boxes.xyxy.cpu().numpy()
        scores = result.boxes.conf.cpu().numpy()
        class_ids = result.boxes.cls.cpu().numpy().astype(int)
        if hasattr(result.boxes, 'id') and result.boxes.id is not None:
            tracker_ids = result.boxes.id.cpu().numpy().astype(int)
        else:
            tracker_ids = np.zeros(len(boxes), dtype=int)
        class_names = result.names
    else:
        boxes = np.empty((0, 4))
        scores = np.empty(0)
        class_ids = np.empty(0, dtype=int)
        tracker_ids = np.empty(0, dtype=int)
        class_names = {}

    detections = sv.Detections(
        xyxy=boxes,
        confidence=scores,
        class_id=class_ids,
        tracker_id=tracker_ids
    )
    detections.class_names = class_names
    return detections


class ObjectDetectionSystem:
    def __init__(self, model_path='yolov8n.pt', tracker="botsort.yaml", 
                 confidence_threshold=0.5, iou_threshold=0.3, memory_duration=5,
                 use_gpu=True, log_dir="object_logs"):
        self.model = YOLO(model_path)
        self.tracker = tracker
        self.confidence_threshold = confidence_threshold
        self.iou_threshold = iou_threshold
        self.logger = ObjectLogger(log_dir)
        self.scene_memory = SceneMemory(memory_duration, confidence_threshold, iou_threshold, self.logger)
        self.device = 0 if use_gpu and cv2.cuda.getCudaEnabledDeviceCount() > 0 else 'cpu'
        self.box_annotator = sv.BoxAnnotator(thickness=2)
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def process_frame(self, frame):
        results = self.model.track(
            frame, 
            conf=self.confidence_threshold,
            iou=self.iou_threshold,
            tracker=self.tracker,
            persist=True,
            device=self.device
        )[0]

        detections = detections_from_yolov8(results)
        missing_objects, new_objects = self.scene_memory.update(detections, frame)
        annotated_frame = self.annotate_frame(frame, detections, missing_objects, new_objects)

        self.frame_count += 1
        elapsed_time = time.time() - self.start_time
        if elapsed_time >= 1.0:
            self.fps = self.frame_count / elapsed_time
            self.frame_count = 0
            self.start_time = time.time()

        cv2.putText(annotated_frame, f"FPS: {self.fps:.1f}", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        return annotated_frame, self.scene_memory.get_status()

    def annotate_frame(self, frame, detections, missing_objects, new_objects):
        annotated_frame = frame.copy()

        if detections is not None and len(detections) > 0:
            labels = [
                f"{detections.class_names[class_id]} {confidence:.2f}"
                for class_id, confidence in zip(detections.class_id, detections.confidence)
            ]
            annotated_frame = self.box_annotator.annotate(
                scene=annotated_frame, 
                detections=detections
            )
            for box, label in zip(detections.xyxy, labels):
                x1, y1, x2, y2 = map(int, box)
                cv2.rectangle(annotated_frame, (x1, y1 - 20), (x1 + 150, y1), (0, 255, 0), -1)
                cv2.putText(annotated_frame, label, (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        for obj in missing_objects:
            x1, y1, x2, y2 = map(int, obj['last_bbox'])
            dash_length = 10
            for i in range(0, int((x2 - x1) * 2 + (y2 - y1) * 2), dash_length * 2):
                if i < (x2 - x1):
                    start_x = x1 + i
                    end_x = min(start_x + dash_length, x2)
                    cv2.line(annotated_frame, (start_x, y1), (end_x, y1), (0, 0, 255), 2)
                    cv2.line(annotated_frame, (start_x, y2), (end_x, y2), (0, 0, 255), 2)
                elif i < (x2 - x1) * 2:
                    j = i - (x2 - x1)
                    start_y = y1 + j
                    end_y = min(start_y + dash_length, y2)
                    cv2.line(annotated_frame, (x1, start_y), (x1, end_y), (0, 0, 255), 2)
                    cv2.line(annotated_frame, (x2, start_y), (x2, end_y), (0, 0, 255), 2)
            cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + 150, y1), (0, 0, 255), -1)
            cv2.putText(annotated_frame, f"MISSING {obj['class_name']}", (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

        for obj in new_objects:
            x1, y1, x2, y2 = map(int, obj['bbox'])
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)
            cv2.rectangle(annotated_frame, (x1, y1 - 25), (x1 + 120, y1), (0, 255, 0), -1)
            cv2.putText(annotated_frame, f"NEW {obj['class_name']}", (x1 + 5, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)

        return annotated_frame


def main():
    parser = argparse.ArgumentParser(description="Object Detection System using BoT-SORT")
    parser.add_argument("--source", type=str, default="0", help="Video source (0 for webcam, or path to video file)")
    parser.add_argument("--model", type=str, default="yolov8n.pt", help="YOLOv8 model path")
    parser.add_argument("--tracker", type=str, default="botsort.yaml", help="Tracker config")
    parser.add_argument("--conf", type=float, default=0.5, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.3, help="IoU threshold")
    parser.add_argument("--memory", type=int, default=5, help="Scene memory duration in seconds")
    parser.add_argument("--output", type=str, default="", help="Output video path")
    parser.add_argument("--log-dir", type=str, default="object_logs", help="Directory for object tracking logs")
    parser.add_argument("--no-gpu", action="store_true", help="Disable GPU acceleration")
    args = parser.parse_args()

    system = ObjectDetectionSystem(
        model_path=args.model,
        tracker=args.tracker,
        confidence_threshold=args.conf,
        iou_threshold=args.iou,
        memory_duration=args.memory,
        use_gpu=not args.no_gpu,
        log_dir=args.log_dir
    )

    if args.source.isdigit():
        cap = cv2.VideoCapture(int(args.source))
    else:
        cap = cv2.VideoCapture(args.source)

    if not cap.isOpened():
        print(f"Error: Could not open video source {args.source}")
        return

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    writer = None
    if args.output:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(args.output, fourcc, fps, (width, height))

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        annotated_frame, status = system.process_frame(frame)

        status_text = f"Tracked: {status['tracked_objects']} | Missing: {status['missing_objects']} | New: {status['new_objects']}"
        cv2.putText(annotated_frame, status_text, (20, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("Object Detection System (BoT-SORT)", annotated_frame)

        if writer:
            writer.write(annotated_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()