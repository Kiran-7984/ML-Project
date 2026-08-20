from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer


def main():

    data = DataTransformation().transform()

    # ModelTrainer().train(data)


if __name__ == "__main__":
    main()