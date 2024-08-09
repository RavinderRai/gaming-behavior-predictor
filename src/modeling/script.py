
import argparse
import os
import json
import pandas as pd
import xgboost as xgb
from sklearn.metrics import accuracy_score, cohen_kappa_score
from pathlib import Path
import joblib
import tarfile



def train(model_directory, train_path, test_path, learning_rate=0.1, max_depth=7,):

    X_train = pd.read_csv(Path(train_path) / "train.csv")
    y_train = X_train[X_train.columns[-1]]
    X_train = X_train.drop(X_train.columns[-1], axis=1)

    X_test = pd.read_csv(Path(test_path) / "test.csv")
    y_test = X_test[X_test.columns[-1]]
    X_test = X_test.drop(X_test.columns[-1], axis=1)

    model = xgb.XGBClassifier(objective='multi:softmax', num_class=3, eval_metric='mlogloss', learning_rate=learning_rate, max_depth=max_depth)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    kappa = cohen_kappa_score(y_test, y_pred)

    model_path = os.path.join(model_directory, "model.joblib")
    joblib.dump(model, model_path)

    with tarfile.open(os.path.join(model_directory, "model.tar.gz"), "w:gz") as tar:
        tar.add(model_path, arcname=os.path.basename(model_path))



if __name__ =='__main__':
    print("[INFO] Extracting arguements")
    parser = argparse.ArgumentParser()

    # hyperparameters sent by the client are passed as command-line arguments to the script.
    parser.add_argument('--learning_rate', type=float, default=0.1)
    parser.add_argument('--max_depth', type=int, default=7)

    args, _ = parser.parse_known_args()

    train(
        model_directory=os.environ["SM_MODEL_DIR"],
        train_path=os.environ["SM_CHANNEL_TRAIN"],
        test_path=os.environ["SM_CHANNEL_TEST"],
        learning_rate=args.learning_rate,
        max_depth=args.max_depth,
    )
