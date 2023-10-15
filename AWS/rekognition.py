import cv2
import boto3
import numpy as np
from time import time

class RekognitionObjectDetection:

    def __init__(self, capture_index, aws_region):
        self.capture = cv2.VideoCapture(capture_index)
        self.rekognition_client = boto3.client('rekognition', region_name=aws_region)
        self.counter = 0  # Initialize counter

    def detect_objects(self, frame):
        # Convert the OpenCV BGR frame to bytes
        _, img_encoded = cv2.imencode('.jpg', frame)
        img_bytes = img_encoded.tobytes()

        # Use Amazon Rekognition to detect objects
        response = self.rekognition_client.detect_labels(
            Image={'Bytes': img_bytes},
            MaxLabels=10,  # Adjust this as needed
            MinConfidence=0.5  # Adjust confidence threshold as needed
        )

        # Extract detected objects and display labels
        labels = [label['Name'] for label in response['Labels']]
        print("Detected Objects:", labels)

        return labels

    def run(self):
        while True:
            start_time = time()
            ret, frame = self.capture.read()
            assert ret

            detected_objects = self.detect_objects(frame)

            # Perform additional processing or actions based on detected objects

            end_time = time()
            fps = 1 / np.round(end_time - start_time, 2)
            cv2.putText(frame, f'FPS: {int(fps)}', (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.imshow('Object Detection with Rekognition', frame)

            if cv2.waitKey(5) & 0xFF == 27:
                break

        self.capture.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    aws_region = 'your_aws_region'  # Replace with your AWS region
    capture_index = 0

    detector = RekognitionObjectDetection(capture_index=capture_index, aws_region=aws_region)
    detector.run()
