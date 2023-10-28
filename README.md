# VXLab Smart Scanning Infrastructure

## Overview

The VXLab Smart Scanning Infrastructure is a project designed to revolutionize resource management and enhance the overall experience within RMIT labs and workshops, with a primary focus on the VXLab. This README provides an overview of the project, the technologies used, and instructions on how to use it.

## Technologies Used

- **Programming Language**: Python
- **Messaging Protocol**: MQTT (Message Queuing Telemetry Transport)
- **Deep Learning Model**: YOLOv8
- **Computer Vision Library**: OpenCV
- **Computer Vision Technology**: OpenPose
- **Cloud Infrastructure**: Amazon Web Services (AWS)
- **Integrated Development Environments**: Visual Studio Code, Android Studio
- **Operating Systems**: Linux-based and Windows
- **Version Control**: GitHub

## How to Use

### Semantic Scanner

The Semantic Scanner, featuring real-time object detection, is implemented using Python and leverages the YOLOv8 machine learning library. To use it:

1. Install the necessary dependencies:
   - PyTorch
   - NumPy
   - OpenCV
   - Ultralytics
   - Paho MQTT
   - Roboflow Supervision

2. Deploy the scanner on a system with a strong GPU and a webcam/video camera for video analysis.

3. Run the any script using the following command:
   ```bash
   python ThomasMQTTSemanticScanner.py

   python BicepCurlMQTT.py

   python objectdetopenposeMQTT.py

   
