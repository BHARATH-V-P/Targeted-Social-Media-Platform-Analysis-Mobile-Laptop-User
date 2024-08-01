# Targeted Social Media Platform Analysis: Laptop and Mobile Users

  This project involves building a classification model to predict weather the product was taken or not. The predictions were seperately made for both Laptop and 
  Mobile Users. The aim of this project
  is to optimize models and create interactive dashboards to visualize trends and specific patterns from the dataset.
  
## Table of Contents
  
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Results](#results)
- [License](#license)

## Introduction

Developing a fine tuned prediction model to determine a customer's decision to purchase a product based on their digital activities from a specific social media platform. The project revolves around a number of variables that are relevant to determining weather the users would be interested in purchasing the product. This targeted marketing is aided by understading the dataset thorugh data visualization using Power Bi and building models tailored to the behavior of the available raw data. Models used include Logistic Regression, SVC, XGB, Random Forest Classifier and LGB. Key  metrics evaluated to produce insightful revelations and optimize the model include roc_auc_score, precision_recall_curve behavior and the confusion matrix of individual models. Both the laptop and mobile-based models have achieved an average accuracy of 98% each.
   
## Installation

1. Clone the repository:
   git clone https://github.com/BHARATH11112222/Targeted-Social-Media-Platform-Analysis-Mobile-Laptop-User

2. Navigate to project directory:
   cd Targeted-Social-Media-Platform-Analysis-Mobile-Laptop-User
   
4. Install the dependencies:
   pip install -r requirements.txt


## Usage

1. Prepare your dataset: Place your dataset in the data/ directory. Ensure it is named Social+Media+Data+for+DSBA (2).csv or adjust the script accordingly.
2. Open the Jupyter Notebook: jupyter notebook "SOCIAL_MEDIA_CLASSIFICATION.ipynb"
   The script will:
   -Preprocess the raw data
   -Train and evaluate models
   -Perform neccessary feature engineering
   -Perform Hyperparameter tuning
   -Evaluate different models based on specific preprocessing done prior to model training.
   -Keep track of specific KPI metrics such as precision_recall curve and rov_curves to understand model behavior for specific preprocessed datasets
3. Review the results:
   Evaluation Metrics: Found in results/evaluation_metrics.txt
   Predictions: Found in results/predictions.csv


## Features
   -Data Preprocessing: Handles missing values, normalization, and encoding.
   -Create different datasets based on two distinct preferred devices 
   -Identifying Outliers to identify characterestics of the extreme values
   -Create 3 different preprocessing pipelines for both datasets.
   -Model Training: Utilizes Logistic Regression, Random Forest Classification, XGBoost , and LGB.
   -Perform necessary hyperparameter tuning using Grid Search CV.
   -Model Evaluation using ROC curve and precision-recall curve
   -Analyse models based on prediction probabilities.

## Results

   Model Performance
   
   Random Forest Classifier (RF)
   ------------------------------------
   Laptop: Best Accuracy of 99.53% and CK Score of 0.9864 using Preprocessing_0.
   Mobile: Best Accuracy of 98.26% and CK Score of 0.9236 using Preprocessing_0.
   
   Logistic Regression (LR)
   ------------------------------------
   Laptop: Best Accuracy of 84.76% and CK Score of 0.5052 using Preprocessing_1.
   Mobile: Best Accuracy of 88.45% and CK Score of 0.3365 using Preprocessing_0.
   
   Support Vector Classifier (SVC)
   ------------------------------------
   Laptop: Best Accuracy of 90.99% and CK Score of 0.7072 using Preprocessing_0.
   Mobile: Best Accuracy of 89.44% and CK Score of 0.3727 using Preprocessing_0.
   
   XGBoost (XGB)
   ------------------------------------
   Laptop: Best Accuracy of 99.53% and CK Score of 0.9866 using Preprocessing_0.
   Mobile: Best Accuracy of 98.96% and CK Score of 0.9555 using Preprocessing_0.
   
   LightGBM (LGB)
   ------------------------------------
   Laptop: Perfect Accuracy of 100% with CK Score of 1.000 using both Preprocessing_0 and Preprocessing_1.
   Mobile: Best Accuracy of 98.11% and CK Score of 0.9177 using Preprocessing_1.
   
   Best Performing Models
   ------------------------------------
   Overall Best: LightGBM on Laptop with 100% Accuracy and CK Score of 1.000.
   Best for Mobile: XGBoost with 98.96% Accuracy and CK Score of 0.9555.
   
   Cross Validation Scores
   ------------------------------------
   Best Mean Accuracy: XGBoost with Mobile Model and Preprocessing_1 (0.9854).
   Best Mean Accuracy for LightGBM: LightGBM with Laptop Model and Preprocessing_1 (0.9857).
   
   Voting Classifier
   ------------------------------------
   Best Accuracy Score: XGB with Laptop Model and Soft Voting (0.9906).
   Best Cohen's Kappa Score: XGB with Laptop Model and Soft Voting (0.9727).
   
   Tuned Random Forest Classifier
   ------------------------------------
   Laptop Model: Accuracy of 98.58%, Precision 1.0, Recall 0.94, F1 Score 0.9691.
   Mobile Model: Accuracy of 98.58%, Precision 0.9965, Recall 0.8378, F1 Score 0.9103.

## License

   This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
