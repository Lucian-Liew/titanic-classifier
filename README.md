# Titanic Survival Classifier

## Overview
A machine learning model built in Python to predict passenger survival on the Titanic dataset. 

## Approach
1. **Data Cleaning:** Handled missing values in the `Age` and `Embarked` columns using median and mode imputation.
2. **Feature Engineering:** Converted categorical variables (`Sex`, `Embarked`) into numerical formats.
3. **Modeling:** Trained a Random Forest Classifier using `scikit-learn`.

## Results
The model achieved an accuracy of **82.68%** on a 20% hold-out test set.

## How to Run
1. Clone this repo.
2. Install dependencies: `pip install pandas scikit-learn`
3. Download `train.csv` from Kaggle and place it in the root folder.
4. Run the script: `python main.py`