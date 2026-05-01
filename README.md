# Diabetes Prediction Using Supervised Machine Learning Models

## Abstract

This project uses supervised machine learning models to predict diabetes risk from clinical measurements in the Pima Indians Diabetes Dataset. We compare interpretable models, ensemble methods, and a neural network to evaluate the trade-off between prediction performance and interpretability. Our preliminary results suggest that neural networks perform slightly best, while logistic regression remains nearly as effective and easier to explain.

---

## Introduction

### Research Topic and Motivation

Diabetes is a major public health problem, and early detection is important because many patients may not be diagnosed until symptoms or complications have already developed. In this project, our main problem is to predict whether a patient has diabetes based on several measurable clinical features. The broader task is closely related to real healthcare decision support: if a model can identify patients at higher risk earlier, it may help doctors prioritize screening, prevention, and follow-up care. This makes the problem interesting as an example of how machine learning can be applied to medical risk prediction.

To approach this problem, we use supervised learning because the dataset includes both input features and a known outcome label indicating whether each patient has diabetes. We compare several models rather than relying on only one method:
    
### Approaches

We use Logistic Regression as a baseline model. It is widely used in healthcare applications because it provides interpretable results, allowing us to understand how each feature contributes to the probability of diabetes. This makes it especially suitable for problems where explanation is as important as prediction.
    
We also include K-Nearest Neighbors (KNN), which takes a fundamentally different approach by classifying observations based on similarity to nearby data points. This method is useful for capturing local patterns in the data and does not assume a specific functional form between features and outcome.
    
In addition, we apply ensemble methods, including Bagging and Random Forest. These models combine multiple decision trees to improve prediction stability. Random Forest, in particular, introduces randomness in feature selection, which can help reduce overfitting and improve generalization. These methods are often effective for structured tabular data and can capture nonlinear relationships.
    
Finally, we explore a Neural Network model, which provides a more flexible framework capable of modeling complex nonlinear interactions between features. Neural networks are widely used in modern machine learning and can potentially achieve strong performance even when relationships between variables are not strictly linear.

The key components of our approach include data preprocessing, feature-based prediction, and systematic comparison across multiple models. We focus on understanding how different modeling assumptions influence prediction performance, rather than relying on a single method. In addition to predictive accuracy, we also consider interpretability as an important factor, especially given the healthcare context of the problem.

### Limitations

There are several limitations to this approach. First, the dataset is relatively small, which may limit the ability of more complex models to fully demonstrate their advantages. Second, the data comes from a specific population, which may reduce the generalizability of the results. Finally, some variables require imputation due to invalid values, which may introduce noise into the modeling process.

---

## Set Up and Data Cleaning


### Dataset Summary

We use the Pima Indians Diabetes Dataset, a binary classification dataset for predicting whether a patient has diabetes.

| Item            | Description                   |
| --------------- | ----------------------------- |
| Observations    | 768                           |
| Features        | 8 numerical clinical features |
| Target variable | `Outcome`                     |
| Classes         | 0 = No diabetes, 1 = Diabetes |

The input features are: `Pregnancies`, `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, `BMI`, `DiabetesPedigreeFunction`, and `Age`.

---

### Basic Statistics and Cleaning

Initial inspection showed that several variables contained invalid zero values. For clinical measurements such as glucose, blood pressure, insulin, and BMI, a value of 0 is not biologically meaningful, so these zeros were treated as missing values.

| Feature                  |  Mean | Notes                             |
| ------------------------ | ----: | --------------------------------- |
| Pregnancies              |  ~3.8 | Right-skewed                      |
| Glucose                  |  ~121 | Strong predictor; 5 invalid zeros |
| BloodPressure            |   ~69 | 35 invalid zeros                  |
| SkinThickness            |   ~20 | 227 invalid zeros                 |
| Insulin                  |   ~80 | 374 invalid zeros; highly skewed  |
| BMI                      |   ~32 | 11 invalid zeros                  |
| DiabetesPedigreeFunction | ~0.47 | Genetic influence                 |
| Age                      |   ~33 | Right-skewed                      |

**Cleaning steps:**

* Replaced invalid zeros in `Glucose`, `BloodPressure`, `SkinThickness`, `Insulin`, and `BMI` with `NaN`
* Filled missing values using the median of each feature
* Verified that no missing values remained after cleaning

---

### Target Distribution and Evaluation Metrics

The dataset shows a moderate class imbalance:

* ~65% non-diabetic (0)
* ~35% diabetic (1)

Because of this imbalance, we evaluate models using multiple metrics:

| Metric   | Purpose                              |
| -------- | ------------------------------------ |
| Accuracy | Overall prediction correctness       |
| ROC-AUC  | Ability to separate classes          |
| F1 Score | Balance between precision and recall |

---

### Exploratory Analysis

A correlation heatmap showed that:

* `Glucose` has the strongest relationship with diabetes outcome
* `BMI` and `Age` are also important predictors
* Some features have moderate correlations, suggesting possible redundancy

This motivated us to explore PCA.

---

### PCA Exploration

We tested PCA as a dimensionality reduction step.

| Step                | Description                            |
| ------------------- | -------------------------------------- |
| Standardization     | All features standardized before PCA   |
| Component selection | Based on cumulative explained variance |
| Result              | Accuracy dropped from ~0.784 to ~0.762 |
| Decision            | PCA not used in final model            |

PCA was not helpful because:

* The dataset only has 8 features
* Features are already meaningful clinically
* PCA removed useful information instead of just noise

---

### Model Setup

We compared several supervised learning models:

| Model               | Role                   | Setup                                                |
| ------------------- | ---------------------- | ---------------------------------------------------- |
| Logistic Regression | Interpretable baseline | Estimates feature effects on diabetes probability    |
| Bagging             | Ensemble baseline      | Combines multiple decision trees                     |
| Random Forest       | Improved ensemble      | Uses feature randomness for better generalization    |
| KNN                 | Distance-based model   | Best K selected via 5-fold CV (K = 55)               |
| Neural Network      | Nonlinear model        | ReLU activation; best: 1 hidden layer with 5 neurons |

All models were evaluated using Accuracy, ROC-AUC, and F1 Score.

---

## Results

### Model Performance Comparison

We evaluated six classification models using Accuracy, ROC-AUC, and F1 Score.

| Model                          | Accuracy | ROC-AUC   | F1 Score |
| ------------------------------ | -------- | --------- | -------- |
| Logistic Regression (baseline) | 0.773    | 0.867     | 0.639    |
| Logistic Regression (Ridge)    | 0.792    | 0.869     | 0.729    |
| Logistic Regression (Lasso)    | 0.792    | 0.868     | 0.729    |
| Bagging                        | 0.818    | 0.874     | 0.767    |
| Random Forest                  | 0.838    | 0.879     | 0.783    |
| Neural Network                 | 0.812    | **0.887** | 0.724    |
| KNN                            | 0.799    | 0.880     | 0.659    |

**Key observations:**

* Random Forest achieved the highest **accuracy (0.838)** and **F1 score (0.783)**
* Neural Network achieved the highest **ROC-AUC (0.887)**
* Regularized logistic regression (Ridge/Lasso) significantly improved over baseline
* All models performed within a relatively narrow ROC-AUC range (0.867–0.887)

---

### Logistic Regression Insights

Regularization improved model performance:

* Accuracy increased from **0.773 → 0.792**
* F1 score increased from **0.639 → 0.729**

Feature importance based on coefficients:

| Feature                  | Coefficient (absolute) |
| ------------------------ | ---------------------- |
| Glucose                  | ~1.10                  |
| BMI                      | ~0.66                  |
| Pregnancies              | ~0.40                  |
| DiabetesPedigreeFunction | ~0.24                  |
| Age                      | ~0.18                  |
| BloodPressure            | ~0.12                  |
| Insulin                  | ~0.04                  |
| SkinThickness            | ~0.00                  |

**Interpretation:**

* Glucose is the strongest predictor of diabetes
* BMI is the second most important factor
* Some features (e.g., SkinThickness) contribute very little

---

### Confusion Matrix Analysis

Comparing logistic models:

* Baseline Logistic Regression had more false negatives (missed diabetes cases)
* Ridge and Lasso reduced false negatives and improved recall for diabetic patients

This shows that **regularization improves sensitivity to the minority class**.

---

### Ensemble Models: Bagging vs Random Forest

| Model         | Accuracy  | ROC-AUC   | F1 Score  |
| ------------- | --------- | --------- | --------- |
| Bagging       | 0.818     | 0.874     | 0.767     |
| Random Forest | **0.838** | **0.879** | **0.783** |

**Interpretation:**

* Random Forest consistently outperformed Bagging
* Feature randomness improves generalization
* Bagging uses all features → more correlated trees
* Random Forest uses feature subsets → better diversity

---

### Feature Importance (Random Forest)

Top important features:

1. Glucose (dominant)
2. BMI
3. Age
4. DiabetesPedigreeFunction

This is consistent with logistic regression results, increasing confidence in findings.

---

### Other Models

#### K-Nearest Neighbors (KNN)

* Best K: 55 (selected via 5-fold cross-validation)
* Accuracy: 0.799
* ROC-AUC: 0.880
* F1 Score: 0.659

#### Neural Network

* Best architecture: 1 hidden layer with 5 neurons
* Accuracy: 0.812
* ROC-AUC: **0.887 (highest)**
* F1 Score: 0.724

**Interpretation:**

* Neural network performs well but is only slightly better than simpler models
* Best architecture is small → suggests underlying relationships are mostly simple

---

## Discussion

### Interpretation of Results

Across all models, ROC-AUC values range from 0.867 to 0.887, indicating a relatively narrow performance range. This suggests that all models capture similar underlying patterns in the data, and that improvements from increased model complexity are limited.

The Neural Network achieved the highest ROC-AUC (0.887), making it the best-performing model. However, its optimal architecture is relatively simple (a single hidden layer with 5 neurons), suggesting that the underlying relationships in the data are not highly complex.

Random Forest provides an overall balance between accuracy and F1 score. This indicates that ensemble methods improve classification performance by reducing variance and handling complex feature interactions effectively.

Logistic Regression, especially with regularization (Ridge and Lasso), remains highly effective while being more interpretable than other models. Its strong performance suggests that simpler, more transparent models can still perform well on this dataset.

---

### Comparison with Existing Approaches

Our results are consistent with previous studies on the Pima Indians Diabetes dataset, where ROC-AUC values typically fall between 0.80 and 0.90. The performance achieved in this project (up to 0.887) is within this expected range, indicating that the models are performing reasonably well.

---

### Consistency of Findings

Across different models, Glucose and BMI consistently appear as the most important features. This agreement across methods increases confidence in the results and aligns with existing medical knowledge about diabetes risk factors.

---

### Limitations

The main limitations include the small dataset size and a limited number of features. These factors may restrict the overall predictive performance.


---

## Conclusion

In this project, we developed and evaluated multiple classification models to predict diabetes using the Pima Indians dataset. We compared Logistic Regression, KNN, Neural Networks, Bagging, and Random Forest using metrics such as accuracy, ROC-AUC, and F1 score.

The Neural Network achieved the highest ROC-AUC, while Random Forest provided the best overall balance between accuracy and F1 score. At the same time, regularized Logistic Regression remained highly competitive and offered greater interpretability.

Overall, the results suggest that while more complex models can provide slight performance improvements, simpler models can achieve comparable results, highlighting an important trade-off between model complexity and interpretability.

These findings are particularly relevant in healthcare applications, where both predictive performance and interpretability are important.

---

## References


* Smith, J. W., Everhart, J. E., Dickson, W. C., Knowler, W. C., & Johannes, R. S. (1988).
  *Using the ADAP learning algorithm to forecast the onset of diabetes mellitus.*
  In *Proceedings of the Annual Symposium on Computer Application in Medical Care* (pp. 261–265).

* Kavakiotis, I., Tsave, O., Salifoglou, A., Maglaveras, N., Vlahavas, I., & Chouvarda, I. (2017).
  *A survey of machine learning and data mining methods for diabetes research.*
  Computational and Structural Biotechnology Journal, 15, 104–116.
  https://doi.org/10.1016/j.csbj.2016.12.005

* Rajendra, P., & Latifi, S. (2021).
  *Prediction of diabetes using logistic regression and ensemble techniques.*
  Computer Methods and Programs in Biomedicine Update, 1, 100032.
  https://doi.org/10.1016/j.cmpbup.2021.100032

* Naz, H., & Ahuja, S. (2020).
  *Deep learning approach for diabetes prediction using PIMA Indian dataset.*
  Journal of Diabetes & Metabolic Disorders, 19, 391–403.
  https://doi.org/10.1007/s40200-020-00520-5

* Abousaber, I., Abdallah, H. F., & El-Ghaish, H. (2025).
  *Robust predictive framework for diabetes classification using optimized machine learning on imbalanced datasets.*
  Frontiers in Artificial Intelligence, 7, 1499530.
  https://doi.org/10.3389/frai.2024.1499530
