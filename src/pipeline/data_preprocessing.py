# from src.components.data_ingestion import DataIngestion
# from src.components.data_transformation import DataTransformation


# def main():

#     # data = DataIngestion().initiate_data_ingestion()

#     DataTransformation().transform()


# if __name__ == "__main__":
#     main()
import pandas as pd
from sklearn.model_selection import train_test_split

def preprocess():

    df = pd.read_csv(
        "data/processed/iri_deterioration.csv"
    )

    y = df["NEXT_MRI"]

    X = df.drop(columns=[
        "NEXT_MRI",
        "NEXT_VISIT_DATE",
        "MRI_CHANGE",
        "IRI_DETERIORATION_RATE",
        "YEARS_TO_NEXT"
    ], errors = "ignore")

    X = X.drop(columns=[
        "STATE_CODE",
        "SHRP_ID",
        "CONSTRUCTION_NO"
    ], errors = "ignore")

    X = pd.get_dummies(
        X,
        drop_first=True
    )

    X = X.fillna(
        X.median(numeric_only=True)
    )

    X = X.fillna(0)

    return train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )


if __name__ == "__main__":
    X_train, X_test, y_train, y_test = preprocess()

    # print("Preprocessing completed!")
    # print("Training data:", X_train.shape)
    # print("Testing data:", X_test.shape)