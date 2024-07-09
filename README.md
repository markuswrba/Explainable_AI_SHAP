# README

LINK TO GITHUB REP: https://github.com/markuswrba/Explainable_AI_SHAP

## Overview

This project involves analyzing and interpreting machine learning models using SHAP explanations. The primary focus is on two datasets: MRI brain scans for tumor classification and clinical data for heart disease prediction. The objective is to compare the interpretability of models trained on image data versus tabular data and investigate how SHAP explanations can enhance understanding of model predictions.

The training process for the models is time-intensive, and as such, the pre-trained models have been stored in Keras format. Due to the large size of these Keras files, they cannot be uploaded directly. For users who wish to explore the notebook, you have two options:

- Train the Models Locally: You can run the provided code on your own machine to train the models.
- Request Pre-trained Models: Alternatively, you can contact the authors to obtain the pre-trained Keras files.

## Datasets

### MRI Brain Tumor Dataset
- **Description:** This dataset contains MRI brain scans categorized into four classes: glioma, meningioma, pituitary tumor, and no tumor.
- **Link:** [Kaggle - Brain Tumor MRI Dataset](https://www.kaggle.com/datasets/masoudnickparvar/brain-tumor-mri-dataset)
- **Usage:** The dataset is used to apply transfer learning on the ResNet50 architecture, a convolutional neural network (CNN) from tensorflow, to classify the brain scans into one of the four categories. SHAP explanations are then applied to interpret the model's predictions.

### Heart Disease Prediction Dataset
- **Description:** This dataset includes clinical and demographic features used to predict the likelihood of heart failure.
- **Link:** [Kaggle - Heart Failure Prediction Dataset](https://www.kaggle.com/datasets/fedesoriano/heart-failure-prediction/data)
- **Usage:** The dataset is used to train an artificial neural net to predict heart disease. Both global and local SHAP explanations are utilized to understand the importance of various features in the model's predictions.

## Project Structure

- **Background and Relevance**
- **MRI Brain Scans Analysis**
  - Data preprocessing and augmentation.
  - Model training and evaluation.
  - SHAP explanations for image data.
- **Heart Disease Prediction:**
  - Data preprocessing and feature engineering.
  - Model training and evaluation.
  - SHAP explanations for tabular data.
- **Analysis and Results**
- **Summary**
- **References**

## Main Dependencies
- Python 
- TensorFlow
- Keras
- scikit-learn
- SHAP
- matplotlib
- seaborn
- numpy
- Pandas

### Contact: 

Markus Wrba: markus.wrba@outlook.com  
Elias Kolarik: e.kolarik@gmx.nat
