import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
from logging import getLogger

from log_code import setup_logging

logger = setup_logging("missing_data_handling") # it will go to log file and run code and run one more log file to output
def missing(X_train, X_test):
    try:
        logger.info(
            f"missing training data before:\n{X_train.isnull().sum()}"
        )

        logger.info(
            f"missing testing data before:\n{X_test.isnull().sum()}"
        )

        for i in X_train.columns:
            train_missing = X_train[i].isnull().sum()
            test_missing = X_test[i].isnull().sum()
            if train_missing > 0:
                logger.info(f"Processing column: {i}")
                # Create new columns containing original values
                X_train[i + "re"] = X_train[i].copy()
                X_test[i + "re"] = X_test[i].copy()
                # Random samples from non-missing values
                t1 = X_train[i].dropna().sample(
                    train_missing,
                    random_state=42
                )

                t2 = X_test[i].dropna().sample(
                    test_missing,
                    random_state=42
                )

                # Change index to missing-value indexes
                t1.index = X_train[X_train[i].isnull()].index
                t2.index = X_test[X_test[i].isnull()].index

                # Fill missing values in the new columns
                X_train.loc[
                    X_train[i].isnull(),
                    i + "re"
                ] = t1

                X_test.loc[
                    X_test[i].isnull(),
                    i + "re"
                ] = t2

                # Drop original column
                X_train = X_train.drop(columns=[i])
                X_test = X_test.drop(columns=[i])

                logger.info(
                    f"Column {i} dropped successfully"
                )

        logger.info(
            f"missing training data after:\n{X_train.isnull().sum()}"
        )

        logger.info(
            f"missing testing data after:\n{X_test.isnull().sum()}"
        )

        return X_train, X_test

    except Exception as e:
        er_ty,er_msg,er_line = sys.exc_info()
        logger.warning(f"error in line: {er_line.tb_lineno} and due to: {er_ty} reason is : {er_msg}")