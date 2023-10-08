import sys

import pandas as pd
import pickle

from src.exception import CustomException
#from src.utils import load_object


class PredictPipline:
    def __init__(self):
        pass

    def load_object(obj,file_path):
        print("loadobjectFunction")
        try:
            with open(file_path,'rb') as file_obj:
                print('loading...')
                return pickle.load(file_obj)
        except Exception as e:
            CustomException(e,sys) 



    def predict(self,features):
        try:
            model_path='Diabetes\model.pkl'
            preprocessor_path='Diabetes\preprocessor.pkl'

            preprocessor=self.load_object(file_path=preprocessor_path)
            
            
            model=self.load_object(file_path=model_path)


            scaled_data=preprocessor.transform(features)
            pred=model.predict(scaled_data)
            return pred
        except Exception as e:
            raise CustomException(e,sys)

class CustomData:
    def __init__(self,
                 gender: str,
                 age: int,
                 hypertension: int,
                 heart_disease: int,
                 smoking_history: str,
                 bmi: int,
                 HbA1c_level: str,
                 blood_glucose_level: int):
        
        self.gender=gender
        self.age=age
        self.hypertension=hypertension
        self.heart_disease=heart_disease
        self.smoking_history=smoking_history
        self.bmi=bmi
        self.HbA1c_level=HbA1c_level
        self.blood_glucose_level=blood_glucose_level

    def get_data_as_data_frame(self):
        
        custom_data_input_dict={
                "gender":[self.gender],
                "age":[self.age],
                "hypertension":[self.hypertension],
                "heart_disease":[self.heart_disease],
                "smoking_history":[self.smoking_history],
                "bmi":[self.bmi],              
                "HbA1c_level":[self.HbA1c_level],
                "blood_glucose_level":[self.blood_glucose_level]

                }
        return pd.DataFrame(custom_data_input_dict)
        
        