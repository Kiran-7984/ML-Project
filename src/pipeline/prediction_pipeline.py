import joblib
import pandas as pd


class PredictionPipeline:

    def __init__(self):

        self.model = joblib.load(
            "models/best_model.pkl"
        )

    def predict(self, data):

        data = pd.DataFrame(data)

        prediction = self.model.predict(
            data
        )

        return prediction


if __name__ == "__main__":

    data = pd.read_csv(
        "data/processed/iri_deterioration.csv"
    )

    data = data.drop(
        columns=["NEXT_MRI","NEXT_VISIT_DATE",
                "MRI_CHANGE",
                "IRI_DETERIORATION_RATE","YEARS_TO_NEXT"],
        errors="ignore"
    )

    predictions = PredictionPipeline().predict(data)

    print(predictions[:10])