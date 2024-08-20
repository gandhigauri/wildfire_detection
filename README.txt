Wildfire Detection
==================

This project contains 5 Python scripts for training, validating, and testing a deep learning model for wildfire detection.

Prerequisites:
- Python 3.7 or higher
- pip (Python package installer)
- CUDA-capable GPU (recommended for faster training if you plan to train your own model)

Instructions:

1. Install Required Packages:
   python install_dependencies.py

2. Set Up Dataset:
   python setup_dataset.py

3. Train model:
   python train_yolo.py --data_yaml path/to/your/data.yaml
   Replace 'path/to/your/data.yaml' with the actual path to your data.yaml file.

4. Validate model:
   python validate_yolo.py --data_yaml path/to/your/data.yaml --model path/to/your/best.pt
   Replace 'path/to/your/data.yaml' with your data.yaml file path and 'path/to/your/best.pt' with the path to your trained model weights.

5. Test model:
   python test_yolo.py --test_data path/to/test/data --model path/to/your/best.pt
   Replace 'path/to/test/data' with the path to your test dataset and 'path/to/your/best.pt' with your model weights path.

Notes:
- You can use the pre-trained model weights that are provided in this folder for validation and testing.
- Ensure all paths are correct before running the scripts.
- You may need to check the path to the train/val/test data in the data.yaml after running setup_dataset.py.