# -*- coding: utf-8 -*-

"""# Task
Analyze the car data from "/content/car data.csv" to build and evaluate a car price prediction model.

## Data loading

### Subtask:
Load the car data from the provided CSV file into a pandas DataFrame.

**Reasoning**:
Import the pandas library, read the CSV file into a DataFrame, and display the first 5 rows.
"""

import pandas as pd

car_data = pd.read_csv('/content/car data.csv')
display(car_data.head())

"""## Exploratory data analysis (eda)

### Subtask:
Perform EDA to understand the data distribution, identify missing values, and visualize relationships between features and the target variable (price).

**Reasoning**:
Display the concise summary and descriptive statistics of the DataFrame, and check for missing values as per instructions 1, 2 and 3.
"""

car_data.info()
display(car_data.describe())
display(car_data.isnull().sum())

"""**Reasoning**:
Visualize the distribution of the target variable 'Selling_Price' and explore the relationships between relevant features and the target variable as per instructions 4 and 5.


"""

import matplotlib.pyplot as plt
import seaborn as sns

plt.figure(figsize=(10, 6))
sns.histplot(car_data['Selling_Price'], kde=True)
plt.title('Distribution of Selling Price')
plt.xlabel('Selling Price')
plt.ylabel('Frequency')
plt.show()

plt.figure(figsize=(15, 10))

plt.subplot(2, 2, 1)
sns.scatterplot(x='Year', y='Selling_Price', data=car_data)
plt.title('Selling Price vs Year')

plt.subplot(2, 2, 2)
sns.scatterplot(x='Driven_kms', y='Selling_Price', data=car_data)
plt.title('Selling Price vs Driven Kms')

plt.subplot(2, 2, 3)
sns.boxplot(x='Fuel_Type', y='Selling_Price', data=car_data)
plt.title('Selling Price vs Fuel Type')

plt.subplot(2, 2, 4)
sns.boxplot(x='Selling_type', y='Selling_Price', data=car_data)
plt.title('Selling Price vs Selling Type')

plt.tight_layout()
plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(1, 2, 1)
sns.boxplot(x='Transmission', y='Selling_Price', data=car_data)
plt.title('Selling Price vs Transmission')

plt.subplot(1, 2, 2)
sns.boxplot(x='Owner', y='Selling_Price', data=car_data)
plt.title('Selling Price vs Owner')

plt.tight_layout()
plt.show()

"""## Data preprocessing

### Subtask:
Handle missing values, encode categorical features, and scale numerical features.

**Reasoning**:
Identify categorical columns and apply one-hot encoding to them. Also, identify numerical columns and apply StandardScaler to them. Although no missing values were found in EDA, include a step to handle them for completeness.
"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import numpy as np

# Identify categorical and numerical columns (excluding the target variable 'Selling_Price')
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
numerical_features = ['Year', 'Present_Price', 'Driven_kms', 'Owner']

# Create transformers for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # handle_unknown='ignore' for unseen categories during prediction

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply the transformations
# Although no missing values were found, we can add a simple imputer step in the pipeline if needed in a general case.
# For this specific dataset, as confirmed by EDA, there are no missing values, so imputation is not strictly necessary.
# However, for a robust pipeline, one might include it.
# Example with a placeholder imputer (not active as no missing values):
# from sklearn.impute import SimpleImputer
# preprocessor = ColumnTransformer(
#     transformers=[
#         ('num', Pipeline([('imputer', SimpleImputer(strategy='mean')), ('scaler', numerical_transformer)]), numerical_features),
#         ('cat', Pipeline([('imputer', SimpleImputer(strategy='most_frequent')), ('onehot', categorical_transformer)]), categorical_features)])

# Apply the preprocessor to the data
# We will apply this when splitting the data into training and testing sets later.
# For now, we can demonstrate the transformation on the full dataset for verification.
# transformed_data = preprocessor.fit_transform(car_data.drop('Selling_Price', axis=1))

# Display the list of transformed features (this will show the original numerical and the new one-hot encoded categorical features)
# This step is for understanding the output of the preprocessor.
# The actual transformation will be done within a pipeline later.
transformed_feature_names = numerical_features + list(preprocessor.named_transformers_['cat'].get_feature_names_out(categorical_features))
print("Transformed features:", transformed_feature_names)

# We are not creating a new dataframe here yet, as the transformation will be part of the model pipeline.
# The preprocessor object is ready to be used in the next steps.

"""**Reasoning**:
The error occurred because `preprocessor.named_transformers_` is only available after `fit` or `fit_transform` has been called. To demonstrate the transformed feature names before fitting, we can manually construct the names based on the original numerical features and the one-hot encoded categorical features.


"""

from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Identify categorical and numerical columns (excluding the target variable 'Selling_Price')
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
numerical_features = ['Year', 'Present_Price', 'Driven_kms', 'Owner']

# Create transformers for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore') # handle_unknown='ignore' for unseen categories during prediction

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Manually construct the transformed feature names based on the original features and expected one-hot encoded output
# This requires knowing the unique categories in the categorical features, which we can get from the original data.
fuel_type_categories = car_data['Fuel_Type'].unique()
selling_type_categories = car_data['Selling_type'].unique()
transmission_categories = car_data['Transmission'].unique()

# Sort categories to ensure consistent ordering for OneHotEncoder
fuel_type_categories.sort()
selling_type_categories.sort()
transmission_categories.sort()


transformed_feature_names = numerical_features + \
                            [f'Fuel_Type_{cat}' for cat in fuel_type_categories] + \
                            [f'Selling_type_{cat}' for cat in selling_type_categories] + \
                            [f'Transmission_{cat}' for cat in transmission_categories]

print("Transformed features:", transformed_feature_names)

# We are not creating a new dataframe here yet, as the transformation will be part of the model pipeline.
# The preprocessor object is ready to be used in the next steps.

"""## Feature engineering

### Subtask:
Create new features that might improve model performance (e.g., age of the car from the year).

**Reasoning**:
Calculate the age of the car and add it as a new column to the dataframe, then display the first few rows.
"""

car_data['Car_Age'] = 2025 - car_data['Year']
display(car_data.head())

"""## Model selection

### Subtask:
Choose appropriate regression models for car price prediction.

**Reasoning**:
Import the selected regression model classes from scikit-learn.
"""

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

print("Chosen models: Linear Regression, Random Forest Regressor")

"""## Model training

### Subtask:
Split the data into training and testing sets and train the selected models.

**Reasoning**:
Split the data into training and testing sets and train the selected models as per the instructions.
"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# Define features (X) and target (y)
X = car_data.drop(['Car_Name', 'Year', 'Selling_Price'], axis=1)
y = car_data['Selling_Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Apply preprocessing
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Instantiate and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_transformed, y_train)

# Instantiate and train Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_transformed, y_train)

print("Models trained successfully.")

"""**Reasoning**:
The error indicates that a column specified in the preprocessor definition ('Year') is not present in the X_train DataFrame. This happened because 'Year' was dropped along with 'Car_Name' and 'Selling_Price' when defining X. The 'Car_Age' column should be used as the numerical feature instead of 'Year'. The preprocessing step needs to be adjusted to reflect this.


"""

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
import numpy as np

# Define features (X) and target (y) - Exclude 'Car_Name', 'Year', and 'Selling_Price'. 'Car_Age' is included.
X = car_data.drop(['Car_Name', 'Year', 'Selling_Price'], axis=1)
y = car_data['Selling_Price']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, shuffle=True)

# Identify categorical and numerical columns (excluding the target variable 'Selling_Price' and 'Year')
categorical_features = ['Fuel_Type', 'Selling_type', 'Transmission']
numerical_features = ['Present_Price', 'Driven_kms', 'Owner', 'Car_Age'] # 'Car_Age' is now a numerical feature

# Create transformers for numerical and categorical features
numerical_transformer = StandardScaler()
categorical_transformer = OneHotEncoder(handle_unknown='ignore')

# Create a column transformer to apply different transformations to different columns
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)])

# Apply preprocessing
X_train_transformed = preprocessor.fit_transform(X_train)
X_test_transformed = preprocessor.transform(X_test)

# Instantiate and train Linear Regression model
lr_model = LinearRegression()
lr_model.fit(X_train_transformed, y_train)

# Instantiate and train Random Forest Regressor model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train_transformed, y_train)

print("Models trained successfully.")

"""## Model evaluation

### Subtask:
Evaluate the trained models using appropriate metrics (e.g., Mean Absolute Error, Mean Squared Error, R-squared).

**Reasoning**:
Import the necessary evaluation metrics and evaluate both trained models using the preprocessed test data and the true test target values. Then print the results for comparison.
"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluate Linear Regression model
lr_predictions = lr_model.predict(X_test_transformed)
lr_mae = mean_absolute_error(y_test, lr_predictions)
lr_mse = mean_squared_error(y_test, lr_predictions)
lr_r2 = r2_score(y_test, lr_predictions)

print("Linear Regression Model Evaluation:")
print(f"Mean Absolute Error (MAE): {lr_mae:.2f}")
print(f"Mean Squared Error (MSE): {lr_mse:.2f}")
print(f"R-squared (R2): {lr_r2:.2f}")
print("-" * 30)

# Evaluate Random Forest Regressor model
rf_predictions = rf_model.predict(X_test_transformed)
rf_mae = mean_absolute_error(y_test, rf_predictions)
rf_mse = mean_squared_error(y_test, rf_predictions)
rf_r2 = r2_score(y_test, rf_predictions)

print("Random Forest Regressor Model Evaluation:")
print(f"Mean Absolute Error (MAE): {rf_mae:.2f}")
print(f"Mean Squared Error (MSE): {rf_mse:.2f}")
print(f"R-squared (R2): {rf_r2:.2f}")

"""## Hyperparameter tuning

### Subtask:
Tune the hyperparameters of the best-performing model to further improve its performance.

**Reasoning**:
Tune the hyperparameters of the Random Forest Regressor model using GridSearchCV to improve its performance.
"""

from sklearn.model_selection import GridSearchCV

# Define a parameter grid for Random Forest Regressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Instantiate GridSearchCV
grid_search = GridSearchCV(estimator=rf_model, param_grid=param_grid,
                           scoring='neg_mean_squared_error', cv=5, n_jobs=-1, verbose=2)

# Fit GridSearchCV to the training data
grid_search.fit(X_train_transformed, y_train)

# Get the best hyperparameters
best_params = grid_search.best_params_
print("Best hyperparameters:", best_params)

# Train a new Random Forest Regressor model with the best hyperparameters
best_rf_model = RandomForestRegressor(**best_params, random_state=42)
best_rf_model.fit(X_train_transformed, y_train)

print("New Random Forest model trained with best hyperparameters.")

"""**Reasoning**:
Evaluate the performance of the new Random Forest model trained with the best hyperparameters.


"""

from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

# Evaluate the best Random Forest model
best_rf_predictions = best_rf_model.predict(X_test_transformed)
best_rf_mae = mean_absolute_error(y_test, best_rf_predictions)
best_rf_mse = mean_squared_error(y_test, best_rf_predictions)
best_rf_r2 = r2_score(y_test, best_rf_predictions)

print("\nRandom Forest Regressor Model with Best Hyperparameters Evaluation:")
print(f"Mean Absolute Error (MAE): {best_rf_mae:.2f}")
print(f"Mean Squared Error (MSE): {best_rf_mse:.2f}")
print(f"R-squared (R2): {best_rf_r2:.2f}")

# Compare with the original Random Forest model
print("\nComparison with Original Random Forest Model:")
print(f"Original RF - MAE: {rf_mae:.2f}, MSE: {rf_mse:.2f}, R2: {rf_r2:.2f}")
print(f"Tuned RF      - MAE: {best_rf_mae:.2f}, MSE: {best_rf_mse:.2f}, R2: {best_rf_r2:.2f}")

"""## Prediction

### Subtask:
Use the trained model to make predictions on new data.

**Reasoning**:
Use the trained best Random Forest model to make predictions on the preprocessed test data and print the first 5 predictions.
"""

# Make predictions on the preprocessed test data
final_predictions = best_rf_model.predict(X_test_transformed)

# Print the first 5 predictions
print("First 5 predictions:")
print(final_predictions[:5])

"""## Summary:

### Data Analysis Key Findings

*   The dataset contains 301 entries and 9 columns with no missing values.
*   The 'Selling\_Price' is right-skewed, with most cars having lower selling prices.
*   Newer cars ('Year') tend to have higher 'Selling\_Price'.
*   'Driven\_kms' shows a weak negative correlation with 'Selling\_Price'.
*   'Fuel\_Type', 'Selling\_type', and 'Transmission' are important predictors of 'Selling\_Price', with diesel cars, dealer sales, and automatic transmissions associated with higher prices.
*   A new feature 'Car\_Age' was engineered, calculated as 2025 minus the 'Year'.
*   Both Linear Regression and Random Forest Regressor models were chosen for prediction.
*   After preprocessing (scaling numerical features and one-hot encoding categorical features), both models were trained.
*   The Random Forest Regressor model performed significantly better than the Linear Regression model on the test set:
    *   Linear Regression: MAE = \$1.22, MSE = \$3.48, R2 = 0.85
    *   Random Forest Regressor: MAE = \$0.63, MSE = \$0.90, R2 = 0.96
*   Hyperparameter tuning of the Random Forest Regressor using GridSearchCV did not improve its performance on the test set, with the best hyperparameters yielding the same evaluation metrics as the original Random Forest model.

### Insights or Next Steps

*   The Random Forest Regressor is the superior model for this dataset based on the evaluation metrics.
*   Further feature engineering or exploring other advanced regression models might potentially improve the prediction accuracy beyond the current Random Forest performance.

"""
