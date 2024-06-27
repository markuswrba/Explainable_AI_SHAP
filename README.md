# README

## Overview

This project involves analyzing and interpreting machine learning models using SHAP explanations. The primary focus is on two datasets: MRI brain scans for tumor classification and clinical data for heart disease prediction. The objective is to compare the interpretability of models trained on image data versus tabular data and investigate how SHAP explanations can enhance understanding of model predictions.

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
