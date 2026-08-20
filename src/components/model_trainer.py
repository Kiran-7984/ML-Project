from pathlib import Path
import pandas as pd
import numpy as np
import joblib

from sklearn.model_selection import GroupShuffleSplit, GroupKFold, GridSearchCV
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from sklearn.metrics import mean_squared_error, r2_score


class ModelTrainer:

    def __init__(self):
        self.model_path = Path("models/best_model.pkl")
        self.results_path = Path("artifacts/model_results.csv")

    def train(self, df):

        y = df["MRI"]
        groups = df["SHRP_ID"] 

        X = df.drop(
            columns=[
                "MRI",
                "SHRP_ID",
                "STATE_CODE",
                "STATE_CODE_EXP"
            ],
            errors="ignore"
        )

        categorical = X.select_dtypes(include="object").columns
        numerical = X.select_dtypes(exclude="object").columns

        preprocessor = ColumnTransformer([
            ("num",
                SimpleImputer(strategy="median"),
                numerical),
            ("cat", Pipeline([
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]),
                categorical
            )
        ])

        splitter = GroupShuffleSplit(test_size=0.2, n_splits=1, random_state=42)
        train_idx, test_idx = next(splitter.split(X, y, groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        groups_train = groups.iloc[train_idx]

        cv = GroupKFold(n_splits=5)

        models = {

            "Linear Regression": (
                LinearRegression(),
                {}
            ),

            "Random Forest": (
                RandomForestRegressor(random_state=42),
                {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [5, 10],
                    "model__min_samples_leaf": [5, 10]
                }
            ),

            "XGBoost": (
                XGBRegressor(random_state=42, objective="reg:squarederror"),
                {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [2, 3],
                    "model__learning_rate": [0.05, 0.1],
                    "model__reg_alpha": [0, 1],
                    "model__reg_lambda": [1, 5]
                }
            )
        }

        results = []
        best_model = None
        best_cv_rmse = float("inf")
        best_name = None

        for name, (model, params) in models.items():

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            grid = GridSearchCV(
                pipeline,
                params,
                cv=cv,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )

            grid.fit(X_train, y_train, groups=groups_train)

            cv_scores = -grid.cv_results_["mean_test_score"][grid.best_index_]
            cv_std = grid.cv_results_["std_test_score"][grid.best_index_]

            prediction = grid.predict(X_test)
            test_rmse = mean_squared_error(y_test, prediction) ** 0.5
            test_r2 = r2_score(y_test, prediction)

            print(f"\n{name}")
            print("Best Parameters:", grid.best_params_)
            print(f"CV RMSE: {cv_scores:.4f}")
            print("RMSE: ", test_rmse)
            print("R2 Score: ", test_r2)

            results.append([name, cv_scores, cv_std, test_rmse, test_r2])

            if cv_scores < best_cv_rmse:
                best_cv_rmse = cv_scores
                best_model = grid.best_estimator_
                best_name = name

        results = pd.DataFrame(
            results,
            columns=["Model", "CV_RMSE", "CV_RMSE_std", "Test_RMSE", "Test_R2"]
        )

        self.results_path.parent.mkdir(exist_ok=True, parents=True)
        self.model_path.parent.mkdir(exist_ok=True, parents=True)

        results.to_csv(self.results_path, index=False)
        joblib.dump(best_model, self.model_path)

        print("\nBest Model:", best_name)
        print(f"Best CV RMSE: {best_cv_rmse:.4f}")
        print("Model saved successfully!")

        return best_model