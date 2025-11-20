#This is where we apply transformation
import sys
import os
from dataclasses import dataclass
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler,OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object
@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('artifacts',"preprocessor.pkl")


class DataTransformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    '''This function returns the preprocessor which needs to be applied'''    
    def get_data_transformer_object(self):
        try:
            num_columns=['writing_score','reading_score']
            cat_columns=['gender','race_ethnicity','parental_level_of_education','lunch','test_preparation_course']

            #Developing a Pipeline 
            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )
            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy="most_frequent")),
                    ("One_hot_encoder",OneHotEncoder())
                ]
            )
            logging.info("Numerical and Categorical Columns are Transformed")

            #Giving the Pipeline to ColumnTransformer
            preprocessor=ColumnTransformer(transformers=
                [
                    ("Numerical_Pipeline",num_pipeline,num_columns),
                    ("Categorical_pipeline",cat_pipeline,cat_columns)
                ]
            )
            return preprocessor
        except Exception as e:
            raise CustomException(e,sys)
    

    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("read train and test data")

            logging.info("Obtaining preprocessing object")
            preprocessor_obj=self.get_data_transformer_object()

            target_column_name="math_score"
            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]
            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]
            logging.info("Applying the preprocessor obj on training and testing")
            input_feature_train_arr=preprocessor_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessor_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]
            test_arr=np.c_[input_feature_test_arr,np.array(target_feature_test_df)]

            logging.info("preprocessed the objs")
            save_object(file_path=self.data_transformation_config.preprocessor_obj_file_path,
                        obj=preprocessor_obj)
            return (train_arr,test_arr,self.data_transformation_config.preprocessor_obj_file_path)


        except Exception as e:
            raise CustomException(e,sys)
            