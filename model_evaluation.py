from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, ElasticNet
from sklearn.svm import SVR
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_models(X, y):
    """Evaluate various regression models on the provided features and target."""
    models = {
        'Random Forest': RandomForestRegressor(random_state=42),
        'Linear Regression': LinearRegression(),
        'Support Vector Regression': SVR(),
        'Gradient Boosting': GradientBoostingRegressor(random_state=42),
        'XGBoost': XGBRegressor(random_state=42),
        'Elastic Net': ElasticNet(random_state=42),
    }
    results = {}
    for model_name, model in models.items():
        cv_scores = cross_val_score(model, X, y, cv=5, scoring='neg_mean_squared_error')
        mean_cv_score = -np.mean(cv_scores)
        model.fit(X, y)
        y_pred = model.predict(X)
        mse = mean_squared_error(y, y_pred)
        r2 = r2_score(y, y_pred)
        results[model_name] = {'MSE': mse, 'R²': r2, 'Mean CV MSE': mean_cv_score}
    return results
