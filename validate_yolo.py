import torch
from yolov5 import val
import argparse
import os

# Parse command-line arguments
parser = argparse.ArgumentParser(description='YOLOv5 Validating')
parser.add_argument('--data_yaml', type=str, required=True, help='Path to data.yaml file')
parser.add_argument('--model', type=str, required=True, help='Path to trained model file')
args = parser.parse_args()

# Set up configuration
data_yaml = args.data_yaml
weights = args.model
img_size = 960
batch_size = 16
conf_thres = 0.3
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Validation
def validate_model():
    val.run(data=data_yaml,
            weights=weights,
            imgsz=img_size,
            batch_size=batch_size,
            device=device, 
            conf_thres=conf_thres,
            project=os.path.dirname(data_yaml),
            name='validation_results')

if __name__ == '__main__':
    # Run validation
    validate_model()