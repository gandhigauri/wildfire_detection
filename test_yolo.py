import torch
from yolov5 import detect
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='YOLOv5 Testing')
parser.add_argument('--test_data', type=str, required=True, help='Path to test data')
parser.add_argument('--model', type=str, required=True, help='Path to trained model file')
args = parser.parse_args()

# Set up configuration
source = args.test_data
weights = args.model
img_size = 960
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Validation
def run_inference():
    detect.run(weights=weights,
            source=source,
            imgsz=img_size,
            device=device,
            project=os.path.dirname(source),
            name='test_results')

if __name__ == '__main__':
    # Run validation
    run_inference()