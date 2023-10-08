import os
import sys

from dataclasses import dataclass

from sklearn.ensemble import RandomForestClassifier

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from sklearn.naive_bayes import GaussianNB

from src.exception import CustomException
from src.logger import logging

from src.utils import save_object,evaluate_model


@dataclass
class ModuleTrainerConfig:
    trained_model_file_path=os.path.join("Diabetes",'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModuleTrainerConfig()


    def initiate_model_trainer(self,train_array,test_array):
        try:
            
            X_train,y_train,X_test,y_test=(
                train_array[:,:-1],
                train_array[:,-1],
                test_array[:,:-1],
                test_array[:,-1]
            )
            logging.info("Train and Test data taken to train model")

            models={
                'Logestic':LogisticRegression(),
                'KNeighbor':KNeighborsClassifier(),
                'DessionTree':DecisionTreeClassifier(),
                'RandomForest':RandomForestClassifier(),
                'NayeBayies':GaussianNB()

            }

            model_report:dict=evaluate_model(X_train=X_train,y_train=y_train,X_test=X_test,y_test=y_test,models=models)

            best_model_score=max(sorted(model_report.values()))

            print(model_report)

            best_model_name=list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
                                                      
            ]

            best_model=models[best_model_name]

            save_object(

                file_path=self.model_trainer_config.trained_model_file_path,
                obj=best_model
                )
            
            predicted=best_model.predict(X_test)

            score=accuracy_score(y_test,predicted)
            return score

        except Exception as e:
            raise CustomException(e,sys)