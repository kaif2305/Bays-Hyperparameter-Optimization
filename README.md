## 🔍 Bayesian Hyperparameter Optimization

### What is Bayesian Optimization?

Bayesian Optimization is a strategy for finding the best hyperparameters of a machine learning model using probability and statistics. Unlike grid or random search, Bayesian optimization builds a probabilistic model of the objective function and uses it to select the most promising hyperparameters to try next.

This method is especially useful when:
- Model training is expensive
- The search space is large or continuous
- Fewer evaluations are desired

---

### How It Works

1. **Model the objective function**  
   A surrogate model (usually a Gaussian Process or Tree-structured Parzen Estimator) approximates the true function mapping hyperparameters to performance (e.g., accuracy).

2. **Select promising hyperparameters**  
   An acquisition function (like Expected Improvement) balances exploration vs. exploitation and suggests the next best hyperparameters to try.

3. **Evaluate and update**  
   The model is trained with those parameters, performance is evaluated, and the surrogate is updated.

This process continues for a predefined number of trials.

---

### Tools Used

In this project, we used **[Optuna](https://optuna.org/)**, a modern and lightweight optimization library that implements Bayesian-style hyperparameter search with TPE (Tree-structured Parzen Estimator).

---

### Example Usage (XGBoost + Optuna)

```python
import optuna
from xgboost import XGBClassifier

def objective(trial):
    params = {
        'max_depth': trial.suggest_int('max_depth', 3, 10),
        'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3),
        'n_estimators': trial.suggest_int('n_estimators', 50, 300),
        'subsample': trial.suggest_float('subsample', 0.6, 1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
        'gamma': trial.suggest_float('gamma', 0, 5),
        'reg_alpha': trial.suggest_float('reg_alpha', 0, 10),
        'reg_lambda': trial.suggest_float('reg_lambda', 0, 10),
        'use_label_encoder': False,
        'eval_metric': 'logloss'
    }

    model = XGBClassifier(**params)
    model.fit(X_train, y_train)
    return model.score(X_valid, y_valid)

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best parameters:", study.best_params)
