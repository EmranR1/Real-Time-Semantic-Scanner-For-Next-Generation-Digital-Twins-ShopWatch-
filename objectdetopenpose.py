import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import mediapipe as mp

from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator

# Initialize Mediapipe components
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

class ObjectPoseDetection:

    def __init__(self, object_capture_index, pose_capture_index):
        self.object_detector = self.initialize_object_detector(object_capture_index)
        self.pose_detector = self.initialize_pose_detector(pose_capture_index)
        self.counter = 0  # Initialize counter
        self.stage = None
        self.prev_wrist_y = 0  # Initialize previous wrist y-coordinate

    def initialize_object_detector(self, capture_index):
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", device)
        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
        model.fuse()
        CLASS_NAMES_DICT = model.model.names
        box_annotator = BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=3, text_scale=1.5)

        return {
            'model': model,
            'CLASS_NAMES_DICT': CLASS_NAMES_DICT,
            'box_annotator': box_annotator,
            'device': device
        }

    def initialize_pose_detector(self, capture_index):
        cap = cv2.VideoCapture(capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        client = None  # You can add MQTT configuration here if needed

        return {
            'cap': cap,
            'client': client
        }

    def detect_objects(self, frame):
        results = self.object_detector['model'](frame)
        xyxys, confidences, class_ids = [], [], []

        # Extract detections for person class
        for result in results[0]:
            class_id = result.boxes.cls.cpu().numpy().astype(int)
            confidence = result.boxes.conf.cpu().numpy()
            print(f"Class ID: {class_id}, Confidence: {confidence}")

            if class_id == 0 and confidence > 0.5:
                xyxys.append(result.boxes.xyxy.cpu().numpy())
                confidences.append(confidence)
                class_ids.append(class_id)

        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        self.labels = [f"{self.object_detector['CLASS_NAMES_DICT'][class_id]} {confidence:0.2f}"
                       for _, confidence, class_id, tracker_id in detections]

        frame = self.object_detector['box_annotator'].annotate(frame=frame, detections=detections, labels=self.labels)

        return frame

    def detect_poses(self, frame):
        with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            # Recolor image to RGB
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(image)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            try:
                landmarks = results.pose_landmarks.landmark
                wrist = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x,
                         landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y]
                wrist_y = wrist[1] * image.shape[0]
                wrist_velocity = abs(wrist_y - self.prev_wrist_y)
                self.prev_wrist_y = wrist_y

                if wrist_velocity > 30:
                    if self.stage != 'down':
                        self.stage = 'down'
                        self.counter += 1
                        print(self.counter)
                        # Publish the counter value to an MQTT topic if needed
                        if self.pose_detector['client'] is not None:
                            self.pose_detector['client'].publish("pose_counter", str(self.counter))
                else:
                    self.stage = 'up'

            except:
                pass

            mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                                      mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
                                      mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2))

            cv2.rectangle(image, (0, 0), (225, 73), (245, 117, 16), -1)
            cv2.putText(image, 'REPS', (15, 12),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            cv2.putText(image, str(self.counter),
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 255, 255), 2, cv2.LINE_AA)

            return image

    def run(self):
        while True:
            start_time = time()
            ret, frame = self.pose_detector['cap'].read()
            assert ret

            frame = self.detect_objects(frame)
            frame = self.detect_poses(frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('Object and Pose Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.pose_detector['cap'].release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detector = ObjectPoseDetection(object_capture_index=0, pose_capture_index=0)
    detector.run()
