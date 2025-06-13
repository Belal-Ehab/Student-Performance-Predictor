# Student Performance Predictor

This project, developed for the Introduction to AI course at New Giza University, predicts students' final grades using regression and classification models. It includes exploratory data analysis (EDA), data cleaning, feature engineering, and model evaluation with 10-fold cross-validation, comparing regression models (Linear Regression, Decision Tree Regressor) and classification algorithms (Naive Bayes, KNN, Logistic Regression, SVM).

## Objectives

- Perform EDA and data cleaning on student performance datasets.
- Engineer features to enhance model performance.
- Develop regression models to predict final grades based on study habits, family background, and past performance.
- Compare classification algorithms for performance prediction using accuracy, F-score, and training time.
- Evaluate models using 10-fold cross-validation.

## Dataset

The project uses the **Student Performance** dataset from the UCI Machine Learning Repository, available at: Student Performance. The dataset includes:

- `student-mat.csv`: Data for 395 students in Mathematics.
- `student-por.csv`: Data for 649 students in Portuguese.
- **Features**: Age, sex, family size, parental education, study time, absences, alcohol consumption, and grades (G1, G2, G3).
- **Target**: Final grade (G3) for regression; binary classification (e.g., pass/fail) based on G3.

## Files

- `Student Performance Predictor.ipynb`: Jupyter notebook containing:
  - Data loading and EDA.
  - Data cleaning and handling of missing values.
  - Feature engineering (e.g., encoding categorical variables, scaling).
  - Regression modeling: Linear Regression, Decision Tree Regressor.
  - Classification modeling: GaussianNB, MultinomialNB, KNN (multiple k values), Logistic Regression, SVM.
  - Model evaluation with RMSE, R², accuracy, F-score, and training time using 10-fold cross-validation.
- `data/`:
  - `student-mat.csv`: Mathematics dataset.
  - `student-por.csv`: Portuguese dataset.

## Setup Instructions

1. **Install Dependencies**:

   ```bash
   pip install pandas numpy scikit-learn matplotlib seaborn jupyter
   ```
2. **Download Datasets**:
   - Place `student-mat.csv` and `student-por.csv` in the `data` folder.
   - Alternatively, download from the UCI link above.
3. **Run the Notebook**:

   ```bash
   jupyter notebook Student\ Performance\ Predictor.ipynb
   ```
   - Ensure file paths in the notebook point to `data/student-mat.csv` and `data/student-por.csv`.
4. **View Results**:
   - The notebook includes graphs for training times and model performance metrics.
   - Conclusions are in the "Analyzing the Result" section.

## Results

- **Best Classification Algorithm**: SVM achieved the highest F-score for both datasets.
- **Training Times**: Visualized in graphs within the notebook.
- **Selected Algorithms**:
  - **Regression**: Linear Regression, due to superior RMSE, R², and training time.
  - **Classification**: SVM, for top or near-top performance across metrics (except time, which is negligible for this dataset but may matter for larger datasets).

## Requirements Fulfilled

- **EDA**: Performed using Pandas, Matplotlib, and Seaborn (e.g., `mat.head()`, `mat.shape`).
- **Data Cleaning**: Handled missing or NaN values in the notebook.
- **Feature Engineering**: Included encoding (e.g., `LabelEncoder`) and scaling (e.g., `StandardScaler`).
- **Classification Comparison**:
  - Algorithms: GaussianNB, MultinomialNB, KNN (multiple k), Logistic Regression, SVM.
  - Metrics: Accuracy, F-score, training time.
  - Method: 10-fold cross-validation (`KFold`).
- **Regression Modeling**: Linear Regression and Decision Tree Regressor, evaluated with RMSE and R².
- **Model Selection**: Justified based on performance and time considerations.
