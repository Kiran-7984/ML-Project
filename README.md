# LTPP IRI Deterioration Prediction

A machine learning project for predicting future pavement condition using LTPP pavement performance, traffic, climate, structural and section-level data.

## Objective

Predict the future MRI (NEXT_MRI) of a pavement section based on its current condition and other pavement-related factors.

## Dataset

The project uses data from the Long-Term Pavement Performance (LTPP) program.

Features include:

- Current MRI
- Climate
- Traffic
- Pavement age
- Pavement layer thickness
- Lane characteristics
- Construction information

## Machine Learning Models

- Linear Regression
- Random Forest
- XGBoost

## Model Selection

GridSearchCV is used for hyperparameter tuning and cross-validation.

Models are compared using:

- RMSE
- R² Score

## Project Pipeline

LTPP Data
↓
Data Ingestion
↓
Data Transformation
↓
Feature Engineering
↓
Train/Test Split
↓
Model Training
↓
GridSearchCV
↓
Best Model
↓
MRI Prediction

## Project Structure

src/
├── components/
│   ├── data_ingestion.py
│   ├── data_transformation.py
│   └── model_trainer.py
│
└── pipeline/
    ├── train_pipeline.py
    └── predict_pipeline.py