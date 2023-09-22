import torch
import numpy as np
import cv2
from time import time
from ultralytics import YOLO
import paho.mqtt.client as mqtt

from supervision.draw.color import ColorPalette
from supervision.tools.detections import Detections, BoxAnnotator

class ObjectDetection:

    def __init__(self, capture_index):
        self.capture_index = capture_index
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        print("Using Device: ", self.device)
        self.model = self.load_model()
        self.CLASS_NAMES_DICT = self.model.model.names
        self.box_annotator = BoxAnnotator(color=ColorPalette(), thickness=3, text_thickness=3, text_scale=1.5)
        self.mqtt_client = mqtt.Client()

    def load_model(self):
        model = YOLO("yolov8n.pt")  # load a pretrained YOLOv8n model
        model.fuse()
        return model

    def predict(self, frame):
        results = self.model(frame)
        return results

    def plot_bboxes(self, results, frame):
        xyxys = []
        confidences = []
        class_ids = []

        for result in results[0]:
            class_ids = result.boxes.cls.cpu().numpy().astype(int)
            confidences = result.boxes.conf.cpu().numpy()

            for class_id, confidence in zip(class_ids, confidences):
                class_name = self.CLASS_NAMES_DICT[class_id]  # Get the class name
                result_str = f"Class ID: {class_id}, Object: {class_name}, Confidence: {confidence}"
                print(result_str)

                # Publish the result to an MQTT topic
                self.mqtt_client.publish("SemanticScanner", result_str)

        detections = Detections(
            xyxy=results[0].boxes.xyxy.cpu().numpy(),
            confidence=results[0].boxes.conf.cpu().numpy(),
            class_id=results[0].boxes.cls.cpu().numpy().astype(int),
        )

        for _, confidence, class_id, tracker_id in detections:
            class_name = self.CLASS_NAMES_DICT[class_id]
            label = f"{class_name} {confidence:0.2f} {class_id}"
            frame = self.box_annotator.annotate(frame=frame, detections=detections, labels=[label])

        return frame

    def __call__(self):
        cap = cv2.VideoCapture(self.capture_index)
        assert cap.isOpened()
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

        # Connect to the MQTT broker
        self.mqtt_client.connect("131.170.250.237", 8080, 60)

        while True:
            start_time = time()
            ret, frame = cap.read()
            assert ret

            results = self.predict(frame)
            frame = self.plot_bboxes(results, frame)

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)

            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('YOLOv8 Detection', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        # Disconnect the MQTT client and cleanup
        self.mqtt_client.disconnect()
        cap.release()
        cv2.destroyAllWindows()

detector = ObjectDetection(capture_index=0)
detector()
