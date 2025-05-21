import pandas as pd
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor, BaggingRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.tree import DecisionTreeRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from scipy.stats import uniform, randint
from sklearn.metrics import mean_absolute_error, mean_squared_error, root_mean_squared_error, r2_score

# Read the Excel file, the 1F throat swab cohort was used as an example
df = pd.read_excel('1F咽拭子.xlsx')

# If it is a radiology examination, replace "分诊签到时间" with "签到/登记时间"
df_rd = df.drop_duplicates(subset=['病人ID', 'Visit times', '分诊签到时间'], keep='first')

# Extract the time information and convert this column to the datetime type. If it is a radiology examination, replace "分诊签到时间" with "签到/登记时间"，replace "采样时间" with "上机检查时间"
df_rd['分诊签到时间'] = pd.to_datetime(df_rd['分诊签到时间'])
df_rd['采样时间'] = pd.to_datetime(df_rd['采样时间'])

# Sort the Excel file in chronological order
df_rd_rank = df_rd.sort_values(by='分诊签到时间', ascending=True)

# Extract the month, day and time information and add it to a new column
df_rd_rank['month'] = df_rd_rank[time_column].dt.month
df_rd_rank['day'] = df_rd_rank[time_column].dt.day
df_rd_rank['hour'] = df_rd_rank[time_column].dt.hour
df_rd_rank['week'] = df_rd_rank[time_column].dt.weekday

# Count the arrival rate, based on "month", "day" and "hour"
grouped = df_rd_rank.groupby(['month', 'day', 'hour']).size().reset_index(name='arrival rate')

# Calculate the number of queuing patient，If it is a radiology examination, replace "分诊签到时间" with "签到/登记时间"，replace "采样时间" with "上机检查时间"
for i in range(len(df_rd_rank)):
    current_sign_in_time = df_rd_rank.loc[i, '分诊签到时间']
    start_index = max(0, i - 10)   # It needs to be dynamically adjusted, ranging from 10 to 100, depending on the congestion level of the queue scene
    count = (df_rd_rank.loc[start_index:i-1, '采样时间'] > current_sign_in_time).sum()
    df_rd_rank.loc[i, 'the number of queuing patient'] = count

# Data preprocessing
y = df["waiting time"]
X = df[["month", "day", "hour", "week", "the number of queuing patient", "arrival rate"]]

# dataset spliting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=0)

# RandomForest
rf = RandomForestRegressor()
rf.fit(X_train, y_train)
rf_predictions = rf.predict(X_test)

# XGBoost
xgb_reg = xgb.XGBRegressor()
xgb_reg.fit(X_train, y_train)
xgb_reg_predictions = xgb_reg.predict(X_test)

# KNN
knn = KNeighborsRegressor()
knn.fit(X_train, y_train)
knn_predictions = knn.predict(X_test)

# MLP
mlp = MLPRegressor()
mlp.fit(X_train, y_train)
mlp_predictions = mlp.predict(X_test)

# SVM
svm = SVR()
svm.fit(X_train, y_train)
svm_predictions = svm.predict(X_test)

# Linear
lr_reg = LinearRegression()
lr_reg.fit(X_train, y_train)
lr_reg_predictions = lr_reg.predict(X_test)

# ElasticNet
en_reg = ElasticNet()
en_reg.fit(X_train, y_train)
en_reg_predictions = en_reg.predict(X_test)

# CART
cart_reg = DecisionTreeRegressor()
cart_reg.fit(X_train, y_train)
cart_reg_predictions = cart_reg.predict(X_test)

# BaggingRegressor
bagging_reg = BaggingRegressor()
bagging_reg.fit(X_train, y_train)
bagging_reg_predictions = bagging_reg.predict(X_test)

# LightGBM
lgb_reg = lgb.LGBMRegressor()
lgb_reg.fit(X_train, y_train)
lgb_reg_predictions = lgb_reg.predict(X_test)

# Model evulation
def evaluate_model(y_test, predictions, model_name):
    mae = mean_absolute_error(y_test, predictions)
    mse = mean_squared_error(y_test, predictions)
    rmse = root_mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    return {"Model": model_name, "MAE": mae, "MSE": mse, "RMSE": rmse, "R2": r2}

# Take LightGBM as an example, using random search to find the optimal hyperparameters
param_dist = {
    'n_estimators': randint(100, 1000),
    'max_depth': randint(3, 20),
    'learning_rate': uniform(0.01, 0.3),
    'subsample': uniform(0.6, 0.4),
    'colsample_bytree': uniform(0.6, 0.4),
    'num_leaves': randint(20, 200),
    'min_child_samples': randint(5, 100),
    'reg_alpha': uniform(0, 1),
    'reg_lambda': uniform(0, 1),
    'min_split_gain': uniform(0, 0.5),
    'bagging_freq': randint(1, 10),
    'feature_fraction': uniform(0.5, 0.5),
    'boosting_type': ['gbdt', 'dart', 'goss']
}

# Initialize the LightGBM regression model
lgb_reg = LGBMRegressor(random_state=0, verbose=-1)

# Use the random search method for hyperparameter adjustment
random_search = RandomizedSearchCV(
    estimator=lgb_reg,
    param_distributions=param_dist,
    n_iter=100,
    cv=5,
    scoring='neg_mean_squared_error',
    random_state=0,
    n_jobs=-1
)

# Fitting data
random_search.fit(X_train, y_train)

# Output the optimal combination of hyperparameters
print(f"Best parameters found: {random_search.best_params_}")

# Make predictions using the model with the optimal parameters
best_lgb_reg = random_search.best_estimator_
lgb_predictions = best_lgb_reg.predict(X_test)
lgb_mae = mean_absolute_error(y_test, lgb_predictions)
lgb_mse = mean_squared_error(y_test, lgb_predictions)
lgb_rmse = root_mean_squared_error(y_test, lgb_predictions)
lgb_r2 = r2_score(y_test, lgb_predictions)
print(f"Mean Absolute Error (MAE): {lgb_mae}")
print(f"Mean Squared Error (MSE): {lgb_mse}")
print(f"Root Mean Squared Error (MAE): {lgb_rmse}")
print(f"R² Score: {lgb_r2}")
