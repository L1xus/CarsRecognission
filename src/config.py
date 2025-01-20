import os
from dotenv import load_dotenv

load_dotenv()

AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")

S3_ROOT = "s3://carset/cars/"
CHECKPOINT_PATH = "resnet34_car_classifier_checkpoint.pth"
NUM_CLASSES = 196
EPOCHS = 10
