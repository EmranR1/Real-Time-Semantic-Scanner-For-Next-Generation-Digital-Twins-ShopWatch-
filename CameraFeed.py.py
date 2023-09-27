import picamera
import picamera.array
import cv2

# Create a VideoCapture object to display the camera feed
cv2.namedWindow("Raspberry Pi Camera Feed", cv2.WINDOW_NORMAL)

# Initialize the camera
with picamera.PiCamera() as camera:
    # Set camera resolution (adjust as needed)
    camera.resolution = (640, 480)
    
    # Create an array to store the video stream
    with picamera.array.PiRGBArray(camera) as stream:
        try:
            for _ in camera.capture_continuous(stream, format="bgr", use_video_port=True):
                # Get the NumPy array from the stream
                frame = stream.array

                # Display the frame in the OpenCV window
                cv2.imshow("Raspberry Pi Camera Feed", frame)
                
                # Clear the stream for the next frame
                stream.truncate(0)

                # Check for the 'q' key to quit the video feed
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

        except KeyboardInterrupt:
            pass

# Release the camera and close the OpenCV window
cv2.destroyAllWindows()
