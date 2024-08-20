from roboflow import Roboflow
import yaml
import os

def update_data_yaml(dataset_location, data_yaml_path):
    # Modify data.yaml to point to correct test, train, and val folders
    with open(data_yaml_path, 'r') as file:
        data = yaml.safe_load(file)

    data['train'] = os.path.join(dataset_location, 'train', 'images')
    data['val'] = os.path.join(dataset_location, 'valid', 'images')
    data['test'] = os.path.join(dataset_location, 'test', 'images')

    with open(data_yaml_path, 'w') as file:
        yaml.dump(data, file, default_flow_style=False)

    print("data.yaml updated successfully.")

def download_dataset(api_key, workspace, project, version):
    print("Downloading dataset from Roboflow...")
    rf = Roboflow(api_key=api_key)
    project = rf.workspace(workspace).project(project)
    dataset = project.version(version).download("yolov5")

    # Get the location of the downloaded dataset
    dataset_location = dataset.location
    print("Dataset downloaded successfully at ", dataset_location)

    data_yaml_path = os.path.join(dataset_location, 'data.yaml')
    # Update data.yaml with relative paths
    update_data_yaml(dataset_location, data_yaml_path)

def main():
    # Roboflow details
    # Enter your roboflow api key here
    ROBOFLOW_API_KEY = "your_api_key_here"
    WORKSPACE = "unlv-c6san"
    PROJECT = "wildfire-detection-with-bounding-boxes"
    VERSION = 6

    # Download
    download_dataset(ROBOFLOW_API_KEY, WORKSPACE, PROJECT, VERSION)

if __name__ == "__main__":
    main()
