# Targeted Social Media Platform Analysis: Laptop and Mobile Users

  This project involves developing a well tuned classification model that predicts whether a certain consumer has purchased the product in question or not. These 
  independent variables are related to factors such as empoyement, the user's social media engagement, the extent of their active role mainly defined by their   
  habits and behaviors such as commenting and liking a particular post on the media platform making their stand on the product noticable.


  
## Table of Contents
  
- [Introduction](#introduction)
- [Installation](#installation)
- [Usage](#usage)
- [Features](#features)
- [Results](#results)
- [License](#license)

## Introduction

Developing a fine tuned prediction model to determine a customer's decision to purchase a product based on their digital activities from a specific social media platform.  The underlying information that helps a tuned model with this decision making are characterized by the predictor variables that define the data record in a multi dimesnional cartesian space. The overall project is developed from a structured dataset which had to be preprocessed and later divided into two different sets of data  points based on whether the users were usign a laptop or a mobile device. The segregation was important due to the difference in login and usage procedures among both. This targeted marketing is aided by understanding the dataset thorugh data visualization using Power Bi and building models tailored to the behavior of the available raw data. Models used include Logistic Regression, SVC, XGB, Random Forest Classifier and LGB. Key  metrics evaluated to produce insightful revelations and optimize the model include roc_auc_score, precision_recall_curve behavior and the confusion matrix of individual models. Both the laptop and mobile-based models have achieved an average accuracy of 98% each. The models where evaluated by thoroghly analysing the roc curve, precision-recall curve and the confusion matrix. 
   
## Installation

1. Clone the repository:
   git clone https://github.com/BHARATH11112222/Targeted-Social-Media-Platform-Analysis-Mobile-Laptop-User

2. Navigate to project directory:
   cd Targeted-Social-Media-Platform-Analysis-Mobile-Laptop-User
   
4. Install the dependencies:
   pip install -r requirements.txt


## Usage

1. Prepare your dataset: Place your dataset in the data/ directory. Ensure it is named Social+Media+Data+for+DSBA (2).csv or adjust the script accordingly.
2. Open the Jupyter Notebook: jupyter notebook "SOCIAL_MEDIA_CLASSIFICATION_MODEL.ipynb"
   The script will:
   - Preprocess the raw data
   - Train and evaluate models
   - Perform neccessary feature engineering
   - Perform Hyperparameter tuning
   - Evaluate different models based on specific preprocessing done prior to model training.
   - Keep track of specific KPI metrics such as precision_recall curve and rov_curves to understand model behavior for specific preprocessed datasets
3. Review the results:
   Evaluation Metrics: Found in results/evaluation_metrics.txt
   Predictions: Found in results/predictions.csv


## Features
   - Data Preprocessing: Handles missing values, normalization, and encoding.
   - Create different datasets based on two distinct preferred devices 
   - Identifying Outliers to identify characterestics of the extreme values
   - Create 3 different preprocessing pipelines for both datasets.
   - Model Training: Utilizes Logistic Regression, Random Forest Classification, XGBoost , and LGB.
   - Perform necessary hyperparameter tuning using Grid Search CV.
   - Model Evaluation using ROC curve and precision-recall curve
   - Analyse models based on prediction probabilities.

## Results

  - Best Performing Models:

    - Overall Best Model: LightGBM on Laptop with Preprocessing_0 and Preprocessing_1, achieving an accuracy of 100% and a CK score of 1.000.
    - Best Model for Mobile: XGBoost with Preprocessing_0, achieving an accuracy of 98.96% and a CK score of 0.9555.
    
  - Cross-Validation Scores:

    - Best Mean Accuracy: XGB with Mobile_model and Preprocessing_1 (0.985423).
    - Best Mean Cohen's Score: LGB with Laptop_model and Preprocessing_1 (0.961498).
    
  - Tuned Random Forest Classifier Performance:

    - Laptop Model: High precision (1.0) with balanced recall and F1 score.
    - Mobile Model: Slightly lower recall compared to precision but still good overall performance.
    
  - Voting Classifier Performance:

    - Best Accuracy: XGB with Laptop_model and Soft Voting (0.990566).
    -  Best Cohen's Kappa Score: Also XGB with Laptop_model and Soft Voting (0.972666)..

## License

   This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
