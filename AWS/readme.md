The Amazon Rekognition.py was developed at the end of the semester and therefore will be included in handover for the next team to attempt. The code remains untested with AWS services however some steps will be provided for guidance. 

Please run command - pip install boto3

Start up steps:


**Set Up AWS Credentials:**

Make sure you have AWS credentials configured, either by setting environment variables, using AWS CLI profiles, or using an IAM role attached to your EC2 instance (if applicable).

**Replace AWS Region:**

Replace 'your_aws_region' with the AWS region where your Rekognition service is located.

**Configure IAM Permissions:**

Ensure that the IAM credentials used by your program have the necessary permissions to access Amazon Rekognition.

**Run the Program:**

Execute the modified program. It captures video frames, sends them to Amazon Rekognition for object detection, and prints the detected object labels.