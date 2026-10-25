# Titanic Survival Prediction

## 1. Project Overview

This project uses machine learning to predict whether a passenger survived the Titanic disaster based on passenger information such as age, gender, passenger class, fare, family details, and port of embarkation.

The project demonstrates the application of data science techniques, including data preprocessing, exploratory data analysis, model training, and performance evaluation.

## 2. Objectives

- Analyze the Titanic passenger dataset.
- Perform exploratory data analysis (EDA) to identify patterns in passenger survival.
- Handle missing values and encode categorical features.
- Build a Logistic Regression classification model.
- Evaluate model performance using standard classification metrics.
- Predict the survival outcome for a new passenger.

## 3. Technologies Used

- **Programming Language:** Python
- **Development Environment:** Jupyter Notebook
- **Data Manipulation:** Pandas, NumPy
- **Data Visualization:** Matplotlib, Seaborn
- **Machine Learning:** Scikit-learn
- **Version Control:** Git and GitHub

## 4. Dataset Description

The Titanic dataset contains information about passengers aboard the Titanic.

The model uses the following input features:

| Feature | Description |
|---|---|
| Pclass | Passenger class |
| Sex | Passenger gender |
| Age | Passenger age |
| SibSp | Number of siblings or spouses aboard |
| Parch | Number of parents or children aboard |
| Fare | Passenger fare |
| Embarked | Port of embarkation |

**Target variable:** `Survived`

- `0` — Passenger did not survive.
- `1` — Passenger survived.

## 5. Methodology

1. Load the dataset using Pandas.
2. Inspect the dataset and identify missing values.
3. Perform exploratory data analysis using visualizations.
4. Separate the input features and target variable.
5. Handle missing numerical and categorical values.
6. Scale numerical features and encode categorical features.
7. Split the dataset into training and testing sets.
8. Train the Logistic Regression model.
9. Generate predictions on the test dataset.
10. Evaluate the model using classification metrics.

## 6. Machine Learning Algorithm

**Logistic Regression**

Logistic Regression is a supervised machine learning algorithm used for classification. It estimates the probability that a passenger belongs to the survived class.

The predicted class is determined using the model's classification threshold.

## 7. Model Evaluation

The model is evaluated using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix
- ROC curve

The actual evaluation results are available in the Jupyter Notebook.

## 8. Project Structure

```text
Titanic-Survival-Prediction/
├── CaseStudy-3.ipynb
├── titanic.csv
├── README.md
├── requirements.txt
└── .gitignore
```

## 9. Installation and Execution

### Prerequisites

- Python 3
- pip
- Jupyter Notebook

### Install dependencies

```bash
pip install -r requirements.txt
```

### Run the notebook

Start Jupyter Notebook:

```bash
jupyter notebook
```

Open `CaseStudy-3.ipynb` and execute the cells in order.

Ensure that `titanic.csv` is in the notebook's working directory.

## 10. Results and Conclusion

The project demonstrates how Logistic Regression can be used to predict passenger survival from historical data.

The model's performance is assessed using multiple evaluation metrics to understand its predictive ability. The results can also help identify relationships between passenger characteristics and survival outcomes.

## 11. Limitations

- The predictions depend on the quality and completeness of the dataset.
- The selected features may not capture every factor that influenced survival.
- The model's performance depends on the train-test split and preprocessing choices.
- Predictions are estimates and should not be interpreted as certainty.

## 12. Author

Developed as part of a Fundamentals of Data Science case study.

## 13. License and Dataset Attribution

Include the dataset's original source and applicable license here. Ensure that the dataset may be redistributed before committing it to a public repository.
