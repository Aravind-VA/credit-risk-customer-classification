'''
in this file we will do feature selection for columns based on techniques for 11 numerical columns using feature selection techniques

'''

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import seaborn as sns
from scipy.stats import yeojohnson
from logging import getLogger
from sklearn.feature_selection import VarianceThreshold
constant_reg = VarianceThreshold(threshold=0.0)
quasi_reg  = VarianceThreshold(threshold=0.1)
from scipy.stats import pearsonr

from log_code import setup_logging
logger = setup_logging("fs")
def best_column(X_train_numerical,X_test_numerical,y_train):
    try:
        logger.info(X_train_numerical.shape)
        logger.info(X_test_numerical.shape)
        logger.info(f" before X_train_numerical: {X_train_numerical.shape}, X_train_numerical : {X_train_numerical.columns}")
        logger.info(f" after X_test_numerical: {X_test_numerical.shape}, X_test_numerical : {X_test_numerical.columns}")
        '''
        implementing constant technique of fetaure selection where variance should be 0 we will remove those columns
        '''
        constant_reg.fit(X_train_numerical)
        logger.info(f"No.of columns are good : {sum(constant_reg.get_support())}") # it will give only count over here
        logger.info(f"No.of columns are bad :{sum(~constant_reg.get_support())}")
        logger.info(f"columns to remove:{X_train_numerical.columns[~constant_reg.get_support()]}")
        X_train_numerical = X_train_numerical.drop(['NPA Status_yeo_trim', 'NumberOfTime30-59DaysPastDueNotWorse_yeo_trim',
       'NumberOfTimes90DaysLate_yeo_trim',
       'NumberOfTime60-89DaysPastDueNotWorse_yeo_trim'],axis=1)
        X_test_numerical = X_test_numerical.drop(['NPA Status_yeo_trim', 'NumberOfTime30-59DaysPastDueNotWorse_yeo_trim',
       'NumberOfTimes90DaysLate_yeo_trim',
       'NumberOfTime60-89DaysPastDueNotWorse_yeo_trim'],axis=1)
        '''
        using constant technique we removed 4 columns as bad for remaining 7 columns we will apply quasi constant technique
        
        '''
        quasi_reg.fit(X_train_numerical)
        logger.info(f"No.of columns are good : {sum(quasi_reg.get_support())}")  # it will give only count over here
        logger.info(f"No.of columns are bad :{sum(~quasi_reg.get_support())}")
        logger.info(f"columns to remove:{X_train_numerical.columns[~quasi_reg.get_support()]}")
        X_train_numerical =  X_train_numerical.drop(['RevolvingUtilizationOfUnsecuredLines_yeo_trim', 'NumberOfDependentsre_yeo_trim'],axis = 1)
        X_test_numerical = X_test_numerical.drop(['RevolvingUtilizationOfUnsecuredLines_yeo_trim','NumberOfDependentsre_yeo_trim'],axis = 1)
        # 2 more columns removed using quasi constant
        logger.info(y_train.shape)
        # below we are doing hypothesis testing
        logger.info(f"{X_train_numerical.columns}")
        values = []
        for i in X_train_numerical.columns:
            values.append(pearsonr(X_train_numerical[i],y_train)) # finding correlation betwen independent and dependent columns
        values = np.array(values) # as values are in list we will convert into array format
        p_values = values[ : , 1]
        s = pd.Series(p_values,index = X_train_numerical.columns)
        plt.figure(figsize=(5,3))
        s.plot.bar()
        plt.show()  # out of 5 columns only 1 column having p value more than 0.05
        X_train_numerical=X_train_numerical.drop(['DebtRatio_yeo_trim'],axis = 1)
        X_test_numerical = X_test_numerical.drop(['DebtRatio_yeo_trim'], axis=1)
        logger.info(X_train_numerical.columns)
        logger.info(X_test_numerical.columns)
        return X_train_numerical, X_test_numerical # it will takes entire code to main file
    except Exception as e:
        er_ty, er_msg, er_line = sys.exc_info()
        logger.warning(f"error in line: {er_line.tb_lineno} and due to: {er_ty} reason is : {er_msg}")