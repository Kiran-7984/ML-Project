import joblib
import pandas as pd


class PredictionPipeline:

    def __init__(self):

        self.model = joblib.load(
            "models/best_model.pkl"
        )

    def predict(self, data):

        data = pd.DataFrame(data)

        data = data.drop(
            columns=[
            "MRI",
            "SHRP_ID",
            "STATE_CODE",
            "STATE_CODE_EXP"
        ],
            errors="ignore"
        )

        prediction = self.model.predict(data)

        return prediction


if __name__ == "__main__":

    data = pd.read_csv(
        "data/processed/iri_deterioration.csv")

    predictions = PredictionPipeline().predict(data)

    print(predictions[:10])

