from health.logger import logging
from health.exception import WeightException
from  health.component.data_transformation import data_transformation
import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.metrics import  accuracy_score,precision_score,recall_score,f1_score
from health.util import evaluate_models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier,AdaBoostClassifier,GradientBoostingClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from catboost import CatBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from health.component.data_transformation import data_transformation
@dataclass

class ModelTrainerConfig:
    train_array,test_array=data_transformation().encoding()

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config=ModelTrainerConfig()
        self.train_array,self.test_array=data_transformation().encoding()

    def initiate_model_trainer(self):
        try:
            logging.info("Split array and test array")
            X_train,y_train,X_test,y_test=self.train_array[:,:-1],self.train_array[:,-1],self.test_array[:,:-1],self.test_array[:,-1]

            models={'Logistic_Regression':LogisticRegression(),
                    'AdaBoost Regressor':AdaBoostClassifier(),
                    'svc':SVC(),
                    'DecisionTreeClassifier':DecisionTreeClassifier(),
                    'GradientBoosting':GradientBoostingClassifier(),
                    'XGBClassifier':XGBClassifier(),
                    'CatBoostClassifier':CatBoostClassifier(),
                    'RandomForest':RandomForestClassifier()}
               

            params={
                
                'RandomForest' : {'n_estimators': [50, 100, 200],'max_depth': [5, 10, 20],'min_samples_split': [2, 5, 10],'min_samples_leaf': [1, 2, 4],'max_features': ['sqrt', 'log2']},
                'Logistic_Regression' :{'C': [0.001, 0.01, 0.1, 1, 10, 100, 1000],'penalty': ['l1', 'l2']},
                'AdaBoost Regressor': {'n_estimators': [50, 100, 200],'learning_rate': [0.1, 0.5, 1.0],'algorithm': ['SAMME', 'SAMME.R']},
                'svc' : {'C': [0.1, 1, 10],'kernel': ['linear', 'poly', 'rbf'],'degree': [2, 3, 4],'gamma': ['scale', 'auto']},
                'DecisionTreeClassifier':{'criterion':['gini','entropy',],'max_depth':[2,4,6,8,10], 'min_samples_split': [2, 4, 6, 8, 10],'min_samples_leaf': [1, 2, 3, 4, 5]},
                'GradientBoosting' : {'n_estimators': [100, 500],'learning_rate': [0.1, 0.5],'max_depth': [3, 5]},
                'XGBClassifier':{'learning_rate': [0.1, 0.01, 0.001],'max_depth': [3, 5, 7],'n_estimators': [50, 100, 200],'subsample': [0.6, 0.8, 1.0],'colsample_bytree': [0.6, 0.8, 1.0],'gamma': [0, 0.1, 0.2],'reg_alpha': [0, 0.1, 0.5],'reg_lambda': [0, 0.1, 0.5]},
                'CatBoostClassifier' : {'iterations': [100, 500, 1000],'learning_rate': [0.01, 0.05, 0.1],'depth': [3, 5, 7]} }
                 
            
            model_report:dict=evaluate_models(X_train,y_train,X_test,y_test, models=models,param=params)
            
             ## To get best model score from dict
            best_model_score = max(sorted(model_report.values()))
            ## To get best model name from dict

            best_model_name = list(model_report.keys())[list(model_report.values()).index(best_model_score)]
            best_model = models[best_model_name]
            
            
            logging.info('i am on k')
            k=[[45,70,1.02,4,3,1,0,1,1,36,53,1.22,115,2.2,15.4,44,7800,5.2,1,0,0,0,1,1,]]
            predicted=best_model.predict(k)


            #accuracy_score1=accuracy_score(y_test,predicted)

            return predicted,self.model_trainer_config.trained_model_file_path,best_model_name,best_model,best_model_score
        
        except Exception as e:
            logging.info(f"the error is {e}")
            raise WeightException(e,sys)
if __name__=="__main__":
    ModelTrainer().initiate_model_trainer()
        

        
            

