# Import necessary libraries
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score
import pickle

# Load the training dataset
train_df = pd.read_csv('data/train.csv')

# Drop columns with a high percentage of missing values
train_df_cleaned = train_df.drop(columns=['PoolQC', 'MiscFeature', 'Alley', 'Fence', 'FireplaceQu', 'Id', 'LotFrontage'])

# Use only the specified 14 features for training
selected_features = [
    'LotArea', 'OverallQual', 'OverallCond', 'YearBuilt', 'YearRemodAdd',
    'GrLivArea', 'FullBath', 'HalfBath', 'BedroomAbvGr', 'KitchenAbvGr',
    'TotRmsAbvGrd', 'Fireplaces', 'GarageCars', 'GarageArea'
]
train_df_selected = train_df_cleaned[selected_features + ['SalePrice']]

# Separate features and target variable
X = train_df_selected.drop(columns=['SalePrice'])
y = train_df_selected['SalePrice']

# Impute missing values with median for numeric features
from sklearn.impute import SimpleImputer
imputer = SimpleImputer(strategy='median')
X_imputed = imputer.fit_transform(X)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_imputed)

# Split the dataset into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Train and evaluate Lasso Regression model
lasso = Lasso(alpha=1.0, random_state=42)
lasso.fit(X_train, y_train)
y_pred_lasso = lasso.predict(X_val)
mse_lasso = mean_squared_error(y_val, y_pred_lasso)
r2_lasso = r2_score(y_val, y_pred_lasso)
mse_lasso_per_datapoint = mse_lasso / len(y_val)

# Train and evaluate Random Forest Regressor
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_pred_rf = rf_model.predict(X_val)
mse_rf = mean_squared_error(y_val, y_pred_rf)
r2_rf = r2_score(y_val, y_pred_rf)
mse_rf_per_datapoint = mse_rf / len(y_val)

# Hyperparameter tuning for Gradient Boosting Regressor
param_grid = {
    'n_estimators': [100, 200, 300],
    'learning_rate': [0.01, 0.1, 0.2],
    'max_depth': [3, 4, 5]
}
gbr = GradientBoostingRegressor(random_state=42)
grid_search = GridSearchCV(estimator=gbr, param_grid=param_grid, cv=3, n_jobs=-1, scoring='r2')
grid_search.fit(X_train, y_train)

# Get the best model from GridSearchCV
gbr_best = grid_search.best_estimator_

# Make predictions with the best Gradient Boosting model
y_pred_gbr = gbr_best.predict(X_val)

# Evaluate the best Gradient Boosting model
mse_gbr = mean_squared_error(y_val, y_pred_gbr)
r2_gbr = r2_score(y_val, y_pred_gbr)
mse_gbr_per_datapoint = mse_gbr / len(y_val)

# Print the results for all models
print("Lasso Regression:\n MSE: {:.2f}, MSE per datapoint: {:.2f}, R^2: {:.3f}".format(mse_lasso, mse_lasso_per_datapoint, r2_lasso))
print("Random Forest Regressor:\n MSE: {:.2f}, MSE per datapoint: {:.2f}, R^2: {:.3f}".format(mse_rf, mse_rf_per_datapoint, r2_rf))
print("Gradient Boosting Regressor (Best Model):\n MSE: {:.2f}, MSE per datapoint: {:.2f}, R^2: {:.3f}".format(mse_gbr, mse_gbr_per_datapoint, r2_gbr))

# Save the best model to a file
with open('housing_price_model.pkl', 'wb') as model_file:
    pickle.dump(gbr_best, model_file)

# Load the saved model
test_df = pd.read_csv('data/test.csv')

# Use only the selected features in the test set
test_df_selected = test_df[selected_features]

# Impute missing values with median for numeric features in the test set
test_imputed = imputer.transform(test_df_selected)

# Standardize the features in the test set
test_scaled = scaler.transform(test_imputed)

# Load the saved model
with open('housing_price_model.pkl', 'rb') as model_file:
    loaded_model = pickle.load(model_file)

# Make predictions on the test set
test_predictions = loaded_model.predict(test_scaled)

# Print the predictions
print("Predictions on the test set:")
print(test_predictions)
