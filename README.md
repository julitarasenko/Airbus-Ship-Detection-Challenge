# Ship Segmentation Project

## Overview 
This project focuses on ship segmentation within maritime images using a U-Net architecture. The goal is to accurately identify and segment ships present in these images for various maritime applications.

## Folder Structure 
- `model/`: Stores the trained model
- `utils/`: Includes utility functions for data preprocessing and model evaluation
- `notebook/`: This folder contains a Jupyter Notebook encompassing the complete pipeline for training, validating, and evaluating the ship segmentation model

## Project Workflow 
1. Data Preparation:
- Loading images and corresponding masks
- Balancing the dataset by creating a balanced training set
2. Model Building:
- Defining a U-Net model for semantic segmentation
- Implementing custom loss functions like Jaccard distance and Dice coefficient
3. Training:
- Training the model using the prepared dataset
- Evaluating model performance using various metrics
4. Inference:
- Generating ship segmentation predictions on new images

## Requirements
- Python 3.10.12
- TensorFlow 2.13.0
- pandas 2.0.3
- scikit-learn 1.2.2

Other dependencies mentioned in `requirements.txt`

## Usage
1. Data Preparation and Model Training:
- Preprocess data and run the training script in `utils/train.py`
- Adjust hyperparameters in the configuration file if necessary
2. Inference:
- Use the trained model to make predictions on new test images in `utils/test.py`

## Acknowledgments
- This project uses a U-Net architecture inspired by research in semantic segmentation
- Parts of the codebase are adapted from various sources and research papers related to image segmentation and deep learning

## Contributors
Yuliia Tarasenko

## License
This project is licensed under the Yuliia Tarasenko License.
