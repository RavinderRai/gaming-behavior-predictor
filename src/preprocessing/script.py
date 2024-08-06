
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split



def preprocess(base_directory):
    """Load the supplied data, split it and transform it."""
    df = _read_data_from_input_csv_files(base_directory)

    # the only transformation we need to do is drop the player id and split the data
    # everything else was done in the etl script
    
    df.drop(columns=['PlayerID'])
    df_train, df_test = train_test_split(df, test_size=0.2)

    y_train = df_train.EngagementLevel
    y_test = df_test.EngagementLevel

    X_train = df_train.drop("EngagementLevel", axis=1)
    X_test = df_test.drop("EngagementLevel", axis=1)

    _save_splits(base_directory, X_train, y_train, X_test, y_test)


def _read_data_from_input_csv_files(base_directory):
    """Read the data from the input CSV files.

    This function reads every CSV file available and
    concatenates them into a single dataframe.
    """
    input_directory = Path(base_directory) / "input"
    files = list(input_directory.glob("*.csv"))

    if len(files) == 0:
        message = f"The are no CSV files in {input_directory.as_posix()}/"
        raise ValueError(message)

    raw_data = [pd.read_csv(file) for file in files]
    df = pd.concat(raw_data)

    # Shuffle the data
    return df.sample(frac=1, random_state=42)


def _save_splits(base_directory, X_train, y_train, X_test, y_test):
    """Save data splits to disk.

    This function concatenates the transformed features
    and the target variable, and saves each one of the split
    sets to disk.
    """
    train = np.concatenate((X_train, y_train), axis=1)
    test = np.concatenate((X_test, y_test), axis=1)

    train_path = Path(base_directory) / "train"
    test_path = Path(base_directory) / "test"

    train_path.mkdir(parents=True, exist_ok=True)
    test_path.mkdir(parents=True, exist_ok=True)

    pd.DataFrame(train).to_csv(train_path / "train.csv", header=False, index=False)
    pd.DataFrame(test).to_csv(test_path / "test.csv", header=False, index=False)


if __name__ == "__main__":
    preprocess(base_directory="/opt/ml/processing")
