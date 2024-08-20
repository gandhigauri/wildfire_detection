import torch
import os
from yolov5 import train
import argparse
import yaml

# Parse command-line arguments
parser = argparse.ArgumentParser(description='YOLOv5 Training')
parser.add_argument('--data_yaml', type=str, required=True, help='Path to data.yaml file')
args = parser.parse_args()

# Set up configuration
hyp = {
    'lr0': 0.01,  # Initial learning rate
    'lrf': 0.01,   # Final learning rate factor 
    'cos_lr':True, # Cosine annealing
    'momentum': 0.937, # SGD momentum/Adam beta1
    'weight_decay': 0.0005, # Optimizer weight decay
    'warmup_epochs': 3.0,  
    'warmup_momentum': 0.8,
    'warmup_bias_lr': 0.1,
    'box': 0.05, # Box loss gain
    'cls': 0.5, # Cls loss gain
    'cls_pw': 1.0, # Cls BCELoss positive_weight
    'obj': 1.0, # Obj loss gain
    'obj_pw': 1.0, # Obj BCELoss positive_weight
    'iou_t': 0.2, # IOU training threshold
    'anchor_t': 4.0, # Anchor-multiple threshold
    'fl_gamma': 2.0, # Focal loss gamma
    'hsv_h': 0.015, # Image HSV-Hue augmentation
    'hsv_s': 0.7, # Image HSV-Saturation augmentation
    'hsv_v': 0.4, # Image HSV-Value augmentation
    'degrees': 0.0, # Image rotation (+/- deg)
    'translate': 0.1, # Image translation (+/- fraction)
    'scale': 0.5, # Image scale (+/- gain)
    'shear': 0.0, # Image shear (+/- deg)
    'perspective': 0.0, # Image perspective (+/- fraction)
    'flipud': 0.0, # Image flip up-down (probability)
    'fliplr': 0.5,  # Image flip left-right (probability)
    'mosaic': 1.0, # Image mosaic (probability)
    'mixup': 0.0, # Image mixup (probability)
    'copy_paste': 0.0, # Segment copy-paste (probability)
}

data_yaml = args.data_yaml
weights = 'yolov5s6.pt'  # Pre-trained weights
img_size = 960
batch_size = 16
epochs = 100
device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

# Training
def train_model():
    # Write hyperparameters to a YAML file
    hyp_path = 'hyp.yaml'
    with open(hyp_path, 'w') as f:
        yaml.dump(hyp, f)
    train.run(data=data_yaml,
              weights=weights,
              imgsz=img_size,
              batch_size=batch_size,
              epochs=epochs,
              device=device,
              multi_scale=True,
              hyp=hyp_path,
              project=os.path.dirname(data_yaml),
              name='yolo_model_output')

if __name__ == '__main__':
    # Run training
    train_model()