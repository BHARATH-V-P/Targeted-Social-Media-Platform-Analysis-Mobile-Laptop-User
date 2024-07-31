# Targeted Social Media Platform Analysis:Laptop and Mobile Users

  This project involves building a classification model to predict weather the product was taken or not. The predictions were seperately made for both Laptop and Mobile Users. The aim of this project
  is to optimize models and create interactive dashboards to visualize trends and specific patterns from the dataset.
  
## Table of Contents
  
- [Introduction](#introduction)F
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Results](#results)
- [License](#license)

## Introduction

   Developing a fine tuned prediction model to determine a customer's decision to purchase a product based on their digital activities from a specific social media platform. The project revolves around a number
   of variables that are relevant to determining weather the users would be interested in purchasing the product. This targeted marketing is aided by understading the dataset thorugh data visualization using 
   Power Bi and building models tailored to the behavior of the available raw data. Models used include Logistic Regression, SVC, XGB, Random Forest Classifier and LGB. Key  metrics evaluated to 
   produce insightful revelations and optimize the model include roc_auc_score, precision_recall_curve behavior and the confusion matrix of individual models. Both the laptop and mobile-based models have achieved
   an average accuracy of 98% each.
   
## Installation

1. Clone the repository:
   git clone https://github.com/BHARATH11112222/Targeted-Social-Media-Platform-Analysis-Mobile-Laptop-User

2. Navigate to project directory:
   cd Targeted-Social-Media-Platform-Analysis-Mobile-Laptop-User
   
4. Install the dependencies:
   pip install -r requirements.txt


## Usage

1. Prepare your dataset: Place your dataset in the data/ directory. Ensure it is named Social+Media+Data+for+DSBA (2).csv or adjust the script accordingly.
2. Open the Jupyter Notebook:
   jupyter notebook "SOCIAL_MEDIA_CLASSIFICATION.ipynb"
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
   -Feature Engineering: Includes permutation importance and feature selection.
   -Model Training: Utilizes Linear Regression, Random Forest Regressors, XGBoost Regressors, and MLP Regressors.
   -Model Stacking: Combines models to enhance performance.
   -Evaluation: Performance metrics and predictions are saved for review.


## Results
   Baseline Model Performance
   -------------------------------
   Linear Regression: R² = 0.930, RMSE = $23,158.28
   Random Forest Regressor: R² = 0.892, RMSE = $28,791.35
   XGBoost Regressor: R² = 0.910, RMSE = $26,231.67
   MLP Regressor: R² = 0.836, RMSE = $35,467.19
   
   Hyperparameter Tuned Models
   --------------------------------
   Random Forest Regressor (Tuned): R² = 0.884, RMSE = $29,859.67
   XGBoost Regressor (Tuned): R² = 0.895, RMSE = $28,426.23
   MLP Regressor (Tuned): R² = 0.931, RMSE = $23,007.31
   
   Feature Engineered Dataset-Based Models
   ----------------------------------------   
   Random Forest Regressor: R² = 0.879, RMSE = $30,426.10
   XGBoost Regressor: R² = 0.900, RMSE = $27,676.39
   MLP Regressor: R² = 0.880, RMSE = $30,304.47
   
   Advanced Models
   ------------------------------------------
   Voting Regressor: RMSE = $23,136.85, MAE = $14,196.87
   Stacking Regressor with RF Meta Model: RMSE = $32,355.92, MAE = $16,893.47
   For detailed metrics, refer to the evaluation_metrics.txt.
   
   -Performance Improvement: R2 score improved by 10% through model stacking.
   -Evaluation Metrics: Detailed in results/evaluation_metrics.txt.
   -Predictions: Available in results/predictions.csv.


## License

   This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
