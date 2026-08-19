from pathlib import Path
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split, GridSearchCV
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

        X = df.drop(columns=["NEXT_MRI"])
        y = df["NEXT_MRI"]

        X = X.drop(
            columns=[
                "SHRP_ID",
                "STATE_CODE",
                "CONSTRUCTION_NO",
                "NEXT_VISIT_DATE",
                "MRI_CHANGE",
                "YEARS_TO_NEXT",
                "IRI_DETERIORATION_RATE"
            ],
            errors="ignore"
        )

        categorical = X.select_dtypes(
            include="object"
        ).columns

        numerical = X.select_dtypes(
            exclude="object"
        ).columns

        preprocessor = ColumnTransformer([
            ("num",
                SimpleImputer(strategy="median"),
                numerical),
            ( "cat",Pipeline([
                    ("imputer",SimpleImputer(strategy="most_frequent")
                    ),
                    ("encoder",
                        OneHotEncoder(
                            handle_unknown="ignore"
                        )
                    )
                ]),
                categorical
            )
        ])

        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=0.2,
            random_state=42
        )

        models = {

            "Linear Regression": (
                LinearRegression(),
                {}
            ),

            "Random Forest": (
                RandomForestRegressor(
                    random_state=42
                ),
                {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [None, 10]
                }
            ),

            "XGBoost": (
                XGBRegressor(
                    random_state=42,
                    objective="reg:squarederror"
                ),
                {
                    "model__n_estimators": [100, 200],
                    "model__max_depth": [3, 5],
                    "model__learning_rate": [0.05, 0.1]
                }
            )
        }

        results = []
        best_model = None
        best_rmse = float("inf")

        for name, (model, params) in models.items():

            pipeline = Pipeline([
                ("preprocessor", preprocessor),
                ("model", model)
            ])

            grid = GridSearchCV(
                pipeline,
                params,
                cv=3,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1
            )

            grid.fit(X_train, y_train)

            prediction = grid.predict(X_test)

            rmse = mean_squared_error(
                y_test,
                prediction
            ) ** 0.5

            r2 = r2_score(
                y_test,
                prediction
            )

            print(f"\n{name}")
            print("Best Parameters:", grid.best_params_)
            print("RMSE:", rmse)
            print("R2 Score:", r2)

            results.append([
                name,
                rmse,
                r2
            ])

            if rmse < best_rmse:
                best_rmse = rmse
                best_model = grid.best_estimator_
                best_name = name

        results = pd.DataFrame(
            results,
            columns=["Model", "RMSE", "R2"]
        )

        self.results_path.parent.mkdir(
            exist_ok=True
        )

        self.model_path.parent.mkdir(
            exist_ok=True
        )

        results.to_csv(
            self.results_path,
            index=False
        )

        joblib.dump(
            best_model,
            self.model_path
        )

        print("\nBest Model:", best_name)
        print("Best RMSE:", best_rmse)
        print("Model saved successfully!")

        return best_model