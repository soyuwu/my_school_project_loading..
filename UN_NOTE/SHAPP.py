import xgboost
import shap
import pre_run
# train an XGBoost model
X_train, X_test, y_train, y_test,length,imbalanceLabel,att= pre_run.dataset("C:/NCKH/CSV/train.csv")# 第五个参数为不平衡
X, y = X_train, y_train
model = xgboost.XGBRegressor().fit(X, y)

# explain the model's predictions using SHAP
# (same syntax works for LightGBM, CatBoost, scikit-learn, transformers, Spark, etc.)
explainer = shap.Explainer(model)
shap_values = explainer(X)

# visualize the first prediction's explanation
shap.plots.waterfall(shap_values[0])