# This is a sample Python script.
from fontTools.merge.util import first


# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    print_hi('PyCharm')
#   python .\main.py this is function we will use to run code in the file
# See PyCharm help at https://www.jetbrains.com/help/pycharm/
# for every python related project we need to use oops method instead of running each and every cell

 #in this file we are going to load data and pass to all ml pipeline to build complete ml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import sklearn
import warnings
warnings.filterwarnings('ignore')
import logging
import sys # here we are importing sys for exception handling and error handling
from log_code import setup_logging
logger = setup_logging("main") # here we are calling function with giving parameter
from sklearn.model_selection import train_test_split
# after splitting the data only we need to modify the data
from missing_data_handling import missing
from vt_outlier import variable_outlier_handling
from fs import best_column
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import OrdinalEncoder
from imblearn.over_sampling import SMOTE # smote is function which do upsampling
from sklearn.preprocessing import StandardScaler # it is nothing but z_score
from all_models import common
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score,confusion_matrix,precision_score,recall_score,f1_score,classification_report
import pickle
class CREDIT_CARD_APPLICATION_DEVELOPMENT:
    try:
        def __init__(self,path):    # this is called as constructor
            self.path = path
            self.df = pd.read_csv(self.path) # reading the dataset
            logger.info(self.df)
            logger.info(self.df.shape) # instead of using print we need to use logging function as files can be interlinked in realtime
            #a=25
            #b=0
            #c = a/b # this will show as error with error name in main.log file
            logger.info(f"before number of null value :{self.df.isnull().sum()}") # when we are using logger function the output won't show in terminal it will directly show in log file
            print(self.df.tail(2)) # as we have print function it will show in terminal
            # here we are using print function to find out idnex number of last 2 rows as we need to delte them
             # after running this code a file will be created in logs file as we created in our project
            self.df = self.df.drop([150000,150001],axis = 0)
            self.df = self.df.drop(["MonthlyIncome.1"],axis = 1)
            logger.info(f"After number of null value :{self.df.isnull().sum()}")
            logger.info("==================before=======================")
            logger.info(self.df["MonthlyIncome"].dtype)
            logger.info(self.df["NumberOfDependents"].dtype) # for no.of dependents we are data type as string which we need to change
            self.df["NumberOfDependents"] = pd.to_numeric(self.df["NumberOfDependents"])
            logger.info("==================after converting datatype=======================")
            logger.info(self.df["MonthlyIncome"].dtype)
            logger.info(self.df["NumberOfDependents"].dtype)
            self.X = self.df.iloc[ : ,:-1]
            self.y = self.df.iloc[ : ,-1]
            self.y = self.y.map({"Good":0,"Bad":1}).astype(int)
            logger.info(f"independent data:{self.X.shape} : \n : {self.X.isnull().sum()}")
            logger.info(self.y.value_counts())
            self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
            logger.info(f"train data size:{self.X_train.shape} : \n : {self.X_train.isnull().sum()}")
            logger.info(f"test data size:{self.X_test.shape} : \n :{self.X_test.isnull().sum()}")


    except Exception as e:
        er_ty,er_msg,er_line = sys.exc_info()
        logger.warning(f"error in line: {er_line.tb_lineno} and due to: {er_ty} reason is : {er_msg}")
    def handling_missing_values(self):
        try:
            # here we are going to use random sample imputation for missing values we dont want to write code here we will write in another file and import
            self.X_train, self.X_test = missing(self.X_train, self.X_test)
            # since we replaced missing values now we to divide columns into numerical and categorical columns
            logger.info(f"total train data:{self.X_train.shape}")
            logger.info(f"total test data:{self.X_test.shape}")
            #logger.info(f"X_train info: {self.X_train.info()}") it is for finding datatype of column
            self.X_train_numerical = self.X_train.select_dtypes(exclude = "str")
            self.X_train_categorical = self.X_train.select_dtypes(include = "str")
            self.X_test_numerical = self.X_test.select_dtypes(exclude = "str")
            self.X_test_categorical = self.X_test.select_dtypes(include = "str")
            logger.info("===================X_train transformation=================")
            logger.info(f"X_train: {self.X_train.shape}")
            logger.info(f"X_train_numerical: {self.X_train_numerical.shape}: \n : {self.X_train_numerical.columns}")
            logger.info(f"X_train_categorical: {self.X_train_categorical.shape} : \n : {self.X_train_categorical.columns}")
            logger.info("===================X_test transformation=================")
            logger.info(f"X_test: {self.X_test.shape}")
            logger.info(f"X_test_numerical: {self.X_test_numerical.shape}: \n : {self.X_test_numerical.columns}")
            logger.info(
                f"X_test_categorical: {self.X_test_categorical.shape} : \n : {self.X_test_categorical.columns}")
            '''
            based on the above code we have training and testing data into numerical and categorical data
            '''

        except Exception as e:
            er_ty,er_msg,er_line = sys.exc_info()
            logger.warning(f"error in line: {er_line.tb_lineno} and due to: {er_ty} reason is : {er_msg}")

    def vt_outlier_handling(self):
        try:
            logger.info(f"X_train_numerical: {self.X_train_numerical.shape}: \n : {self.X_train_numerical.columns}")
            logger.info(f"X_test_numerical: {self.X_test_numerical.shape}: \n : {self.X_test_numerical.columns}")
            self.X_train_numerical,self.X_test_numerical=variable_outlier_handling(self.X_train_numerical,self.X_test_numerical) # for this now we need to send variable
            logger.info(f"X_train_numerical: {self.X_train_numerical.shape}: \n : {self.X_train_numerical.columns}")
            logger.info(f"X_test_numerical: {self.X_test_numerical.shape}: \n : {self.X_test_numerical.columns}")
        except Exception as e:
            er_ty,er_msg,er_line = sys.exc_info()
            logger.warning(f"error in line: {er_line.tb_lineno} and due to: {er_ty} reason is : {er_msg}")

    def feature_selection(self): # we need to call this column also
        try:
            logger.info(f"before feature selection: {self.X_train_numerical.shape} : \n : {self.X_test_numerical.shape}")
            self.X_train_numerical,self.X_test_numerical=best_column(self.X_train_numerical,self.X_test_numerical,self.y_train) # itwill call function from fs file
            logger.info(f"After feature selection: {self.X_train_numerical.shape} : \n : {self.X_test_numerical.shape}")
        except Exception as e:
            er_ty,er_msg,er_line = sys.exc_info()
            logger.warning(f"error in line: {er_line.tb_lineno} and due to: {er_ty} reason is : {er_msg}")
    def cat_to_numerical(self):
        try:
            logger.info(f"categorical columns: {self.X_train_categorical.columns}")
            '''
            apply nominal transformation to categorical data i.e one hot encoding on gender nd region column
            '''
            one_hot_obj = OneHotEncoder(drop = "first")
            one_hot_obj.fit(self.X_train_categorical[["Gender","Region"]])
            values_1 = one_hot_obj.transform(self.X_train_categorical[["Gender","Region"]]).toarray()
            values_2 = one_hot_obj.transform(self.X_test_categorical[["Gender","Region"]]).toarray()
            ## converting all the derived values into seperate dataframe
            t1 = pd.DataFrame(data=values_1,columns = one_hot_obj.get_feature_names_out())
            t2 = pd.DataFrame(data=values_2,columns = one_hot_obj.get_feature_names_out())
            self.X_train_categorical.reset_index(drop = True, inplace = True)
            t1.reset_index(drop = True, inplace = True)
            self.X_test_categorical.reset_index(drop = True, inplace = True)
            t2.reset_index(drop = True, inplace = True)
            self.X_train_categorical_new = pd.concat([self.X_train_categorical, t1], axis = 1) # we are creating new table using both values
            self.X_train_categorical_new = self.X_train_categorical_new.drop(["Gender","Region"], axis = 1)
            self.X_test_categorical_new = pd.concat([self.X_test_categorical, t2], axis = 1)
            self.X_test_categorical_new = self.X_test_categorical_new.drop(["Gender","Region"], axis = 1)
            '''
            now we will implement ordinal encoding for the remaining columns where priority of order matters
            
            '''
            odi_obj = OrdinalEncoder()
            odi_obj.fit(self.X_train_categorical[['Rented_OwnHouse', 'Occupation', 'Education']])
            values_3 = odi_obj.transform(self.X_train_categorical[['Rented_OwnHouse', 'Occupation', 'Education']]) # here transofrm is applying in the same columns so we are not using toarray conversion
            values_4 = odi_obj.transform(self.X_test_categorical[['Rented_OwnHouse', 'Occupation', 'Education']])

            t3 = pd.DataFrame(data=values_3,columns = odi_obj.get_feature_names_out()+"_odinal")
            t4 = pd.DataFrame(data = values_4,columns = odi_obj.get_feature_names_out()+"_odinal")

            t3.reset_index(drop = True, inplace = True)
            t4.reset_index(drop = True, inplace = True)

            self.X_train_categorical_new.reset_index(drop = True, inplace = True)
            self.X_test_categorical_new.reset_index(drop = True, inplace = True)
            '''
            now we need to concat odinal table values categorical new table for both train and test
            '''
            self.X_train_categorical_new = pd.concat([self.X_train_categorical_new, t3], axis = 1)
            self.X_test_categorical_new = pd.concat([self.X_test_categorical_new, t4], axis = 1)

            self.X_train_categorical_new = self.X_train_categorical_new.drop(['Rented_OwnHouse', 'Occupation', 'Education'], axis = 1)
            self.X_test_categorical_new = self.X_test_categorical_new.drop(['Rented_OwnHouse', 'Occupation', 'Education'], axis = 1)
            logger.info("after converting categorical values to numerical")
            logger.info(f"X_train_cat_to_numercial: {self.X_train_categorical_new.shape}: \n : {self.X_train_categorical_new.columns}: \n : {self.X_train_categorical_new.isnull().sum()}")
            logger.info(f"X_test_cat_to_num:{self.X_test_categorical_new.shape}: \n : {self.X_test_categorical_new.columns}: \n : {self.X_test_categorical_new.isnull().sum()}")
            '''
            finally we are combining numerical and categorical values into one table
            
            '''
            # for that first we need to reset index for both tables
            self.X_train_numerical.reset_index(drop = True, inplace = True)
            self.X_train_categorical_new.reset_index(drop = True, inplace = True)
            self.X_test_numerical.reset_index(drop = True, inplace = True)
            self.X_test_categorical_new.reset_index(drop = True, inplace = True)

            self.final_training_data = pd.concat([self.X_train_numerical, self.X_train_categorical_new], axis = 1)
            self.final_testing_data = pd.concat([self.X_test_numerical, self.X_test_categorical_new], axis = 1)

            logger.info("======== final dataset =========")
            logger.info(f"final train independent dataset : {self.final_training_data.shape}: \n : {self.final_training_data.columns}: \n : {self.final_training_data.isnull().sum()}")
            logger.info(f"final test independent dataset: {self.final_testing_data.shape}: \n : {self.final_testing_data.columns}: \n : {self.final_testing_data.isnull().sum()}")
            '''
            we need to see now does the data has been balanced or not for testing
            '''
        except Exception as e:
            er_ty,er_msg,er_line = sys.exc_info()
            logger.warning(f"error in line: {er_line.tb_lineno} and due to: {er_ty} reason is : {er_msg}")
    def balance_data(self):
        try:
            logger.info("=======  before checking data is balanced or not==========")
            logger.info(f"no of good customers:{0} -> : {sum(self.y_train==0)}")
            logger.info(f"no of bad  customers:{1} -> : {sum(self.y_train == 1)}")
            '''
            as data is very biased we need to make data very balanced
            we will solve this discrepancies using over sampling(up sampling) or under sampling(down sampling
            up sampling increasing the unblanaced outcomes values
            '''
            smote_reg =SMOTE(random_state=42)
            self.final_training_data_up,self.y_train_up=smote_reg.fit_resample(self.final_training_data,self.y_train)
            logger.info("=======  after checking data is balanced or not==========")
            logger.info(f"no of good customers:{0} -> : {sum(self.y_train_up == 0)}")
            logger.info(f"no of bad  customers:{1} -> : {sum(self.y_train_up == 1)}")
        except Exception as e:
            er_ty,er_msg,er_line = sys.exc_info()
            logger.warning(f"error in line: {er_line.tb_lineno} and due to: {er_ty} reason is : {er_msg}")



    def scaling_data(self): # here we will scale down all the values of independent data
        try:
            #print(self.final_training_data)
            sc = StandardScaler() # as standard scaler is a class we will object for that
            sc.fit(self.final_training_data_up) # fit will do all the maths and transform will trasform all the values
            self.final_training_data_up_scaled=sc.fit_transform(self.final_training_data_up)
            self.final_testing_data_up_scaled = sc.transform(self.final_testing_data)
           # print(self.final_testing_data_up_scaled) # here the data will be given in array form as we scale down
            #common(self.final_training_data_up_scaled,self.y_train_up,self.final_testing_data_up_scaled,self.y_test)
            # after running the above model we will get to know which model is working better so we making it as comment
            logger.info("======== training the logistic regression model ==========")
            self.reg = LogisticRegression(random_state=42)
            self.reg.fit(self.final_training_data_up_scaled,self.y_train_up)
            self.y_test_predictions = self.reg.predict(self.final_testing_data_up_scaled)
            logger.info(f"confusion matrix: {confusion_matrix(self.y_test,self.y_test_predictions)}") # it is for finfing accuracy score
            logger.info(f"classification report: {classification_report(self.y_test, self.y_test_predictions)}")
            logger.info(f"test accuracy: {accuracy_score(self.y_test, self.y_test_predictions)}")
            logger.info("======= saving the scaled and logistic regression model into pickle file ==========")
            with open("standard_scaler.pkl","wb") as f:
                pickle.dump(sc,f)


            with open("credit_card.pkl","wb") as f1: # here we will create logistic regression model file
                pickle.dump(self.reg,f1)



        except Exception as e:
            er_ty,er_msg,er_line = sys.exc_info()
            logger.warning(f"error in line: {er_line.tb_lineno} and due to: {er_ty} reason is : {er_msg}")







if __name__ == '__main__':
    try:
        obj = CREDIT_CARD_APPLICATION_DEVELOPMENT("creditcard.csv") #if we call this the constructor will response
        obj.handling_missing_values() # to get function whatever is mentioned under that function
        obj.vt_outlier_handling()
        obj.feature_selection()
        obj.cat_to_numerical()
        obj.balance_data()
        obj.scaling_data()
    except Exception as e:
        er_ty,er_msg,er_line = sys.exc_info()
        logger.warning(f"error in line: {er_line.tb_lineno} and due to: {er_ty} reason is : {er_msg}")
