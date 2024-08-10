# Predicting Gaming Behavior with AWS

This repository contains an end-to-end AWS data science project, predicting gaming behavior with an ML model. More specifically, the problem is predicting engagement levels of gamers. In this case, the dataset is already relatively clean, so we didn't need to do a whole lot. The priority is mostly in some basic data cleaning, feature engineering, and then training the model. To complete this project, you just need to have AWS CLI installed. We will take data from an S3 bucket, clean it with AWS Glue, then run a small training pipeline to train a model on our dataset. You can find the original dataset here: [Game Behavior Dataset](https://www.kaggle.com/datasets/rabieelkharoua/predict-online-gaming-behavior-dataset).

## Exploration

Exploring the data was quick and can be found in the exploration.ipynb file in the notebooks folder. Here we took a close look at the data, defined the cleaning and preprocessing steps we would need to take locally, and then ran grid search to train an XGBoost model.

## AWS Deployment

The second notebook called aws_deployment.ipynb takes the exploration steps and pushes python scripts for an ETL job, Preprocessing job, and training job, to AWS. The ETL job is done with AWS Glue, and the preprocessing and training jobs are done as a sequential pipeline in sagemaker. This brings up the end-to-end part, as we do a bit of data engineering and data science with MLOps best practices. Finally, the model is tested locally in that notebook.
