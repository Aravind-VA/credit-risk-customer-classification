import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from scipy.stats import yeojohnson
from logging import getLogger

from log_code import setup_logging

logger = setup_logging("vt_outlier")

def variable_outlier_handling(X_train_numerical,X_test_numerical):
    try:
        logger.info(f"X_train data:{X_train_numerical.shape} :\n : {X_train_numerical.columns} ")
        logger.info(f"X_test data:{X_test_numerical.shape} :\n : {X_test_numerical.columns} ")
        # for finding all the columns are in normal distribution or not
        '''
        Apply yeo johnson outlier detection and update it into new columns and delete old columns
        '''
        for k in X_train_numerical.columns:
            X_train_numerical[k+"_yeo"],lam_value =yeojohnson(X_train_numerical[k])
            X_test_numerical[k+"_yeo"],lam_value = yeojohnson(X_test_numerical[k])
            ## after creating these new columns we will remove old columns so we can apply outliers tranformation over new columns
            X_train_numerical = X_train_numerical.drop([k], axis = 1)
            X_test_numerical = X_test_numerical.drop([k], axis = 1)

            iqr = X_train_numerical[k+"_yeo"].quantile(0.75)-X_test_numerical[k+"_yeo"].quantile(0.25)
            lower_limit = X_train_numerical[k+"_yeo"].quantile(0.25)-(iqr*1.5)
            upper_limit = X_test_numerical[k+"_yeo"].quantile(0.75)+(iqr*1.5)

            X_train_numerical[k+"_yeo_trim"] = np.where(X_train_numerical[k+"_yeo"]< lower_limit, lower_limit,
                     np.where(X_train_numerical[k+"_yeo"] > upper_limit, upper_limit,X_train_numerical[k+"_yeo"]))
            X_test_numerical[k+"_yeo_trim"] = np.where(X_test_numerical[k+"_yeo"]< lower_limit, lower_limit,
                                                      np.where(X_test_numerical[k+"_yeo"]>upper_limit, upper_limit,X_test_numerical[k+"_yeo"]))
            X_train_numerical = X_train_numerical.drop([k+"_yeo"], axis = 1)
            X_test_numerical = X_test_numerical.drop([k+"_yeo"], axis = 1)







        #plt.figure(figsize=(5, 3))
        #for i in X_test_numerical.columns:
            #sns.boxplot(x = X_train_numerical[i])
            #plt.title(f"{i}")
            #plt.show() once we are initiating the file we need to make this block of code uncomment
        logger.info(f" After_X_train data:{X_train_numerical.shape} :\n : {X_train_numerical.columns} ")
        logger.info(f"After_X_test data:{X_test_numerical.shape} :\n : {X_test_numerical.columns} ")
        return X_train_numerical, X_test_numerical #it returns to main file over here


    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.warning(f"error in line: {er_line.tb_lineno} and due to: {er_ty} reason is : {er_msg}")