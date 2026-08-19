from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation


def main():

    # data = DataIngestion().initiate_data_ingestion()

    DataTransformation().transform()


if __name__ == "__main__":
    main()