import sys
from dataclasses import dataclass
from src.utils import save_object

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer # create pipline 
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

from sklearn.preprocessing import StandardScaler,OneHotEncoder,OrdinalEncoder


from src.exception import CustomException
from src.logger import logging

import os

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path=os.path.join('Diabetes','preprocessor.pkl')

class Datatranformation:
    def __init__(self):
        self.data_transformation_config=DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            numerical_columns=['age','hypertension','heart_disease','bmi','HbA1c_level','blood_glucose_level']
            caterogical_columns=['gender','smoking_history']

            num_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='mean')),
                    ("scaler",StandardScaler(with_mean=False))
            ]
            )
            logging.info("Numerical columns encoding completed")

            cat_pipeline=Pipeline(
                steps=[
                    ("imputer",SimpleImputer(strategy='most_frequent')),
                    ("Label_encoder",OrdinalEncoder()),
                    ("scaler",StandardScaler(with_mean=False))

                ]

            )
 
            logging.info("Categorical columns encoding completed")

            preprocessor=ColumnTransformer(
                    [
                        ("num_pipline",num_pipeline,numerical_columns),
                        ("cat_pipline",cat_pipeline,caterogical_columns)
                    ]

            )
            return preprocessor

            


        except Exception as e:
            raise CustomException(e,sys)
        
        


    def initiate_data_transformation(self,train_path,test_path):
        try:
            train_df=pd.read_csv(train_path)
            test_df=pd.read_csv(test_path)

            logging.info("train and test data imported to perform data  transformation")

            preprocessing_obj=self.get_data_transformer_object()

            target_column_name="diabetes"


            input_feature_train_df=train_df.drop(columns=[target_column_name],axis=1)
            target_feature_train_df=train_df[target_column_name]

            input_feature_test_df=test_df.drop(columns=[target_column_name],axis=1)
            target_feature_test_df=test_df[target_column_name]

            logging.info("Transforming data is started")

            input_feature_train_arr=preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr=preprocessing_obj.transform(input_feature_test_df)

            train_arr=np.c_[
                input_feature_train_arr,np.array(target_feature_train_df)
            ]


            test_arr=np.c_[
                input_feature_test_arr,np.array(target_feature_test_df)
            ]

            logging.info("Saved preprocessing object")

            save_object(

                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )


            return train_arr,test_arr

        except Exception as e:
            raise CustomException(e,sys)
