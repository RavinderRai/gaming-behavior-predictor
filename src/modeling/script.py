
from sklearn.ensemble import  RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score
import sklearn
import joblib
import boto3
import pathlib
from io import  StringIO
import argparse
import os
import json
import numpy as np
import pandas as pd

def model_fn(model_dir):
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ =='__main__':

    print("[INFO] Extracting arguements")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--n_estimators', type=int, default=100)
    parser.add_argument('--random_state', type=int, default=0)

    # Data, model, and output_directories
    parser.add_argument('--model-dir', type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument('--train', type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument('--test', type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument('--train-file', type=str, default='train-V-1.csv')
    parser.add_argument('--test-file', type=str, default='test-V-1.csv')

    args, _ = parser.parse_known_args()

    print("SKLearn Version: ", sklearn.__version__)
    print("Joblib Version: ", joblib.__version__)

    print("[INFO] Reading data")
    print()
    df_train = pd.read_csv(os.path.join(args.train, args.train_file))
    df_test = pd.read_csv(os.path.join(args.test, args.test_file))

    features = list(df_train.columns)
    label = features.pop(-1)

    print("Building training and testing datasets")
    X_train = df_train[features]
    X_test = df_test[label]
    y_train = df_train[label]
    y_test - df_test[label]

    print("Training model")
    model = XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', learning_rate=0.1, max_depth=7)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    model_path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, model_path)
    print("Model persisted at ", model_path)
    print()
    print("--- METRIC RESULTS ---")
    print("[TESTING] Model Accuracy is: ", accuracy)
    print("[TESTING] Kappa Score is: ", kappa)
