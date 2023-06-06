import numpy as np
import pandas as pd
import sys
from health.exception import WeightException
from health.logger import logging
from sklearn.model_selection import train_test_split



class data_ingestion:
    def __init__(self):
        pass
    def data_ingestion1(self):
        try:
            
            df=pd.read_csv(r'E:\weight\health\artifact\ObesityDataSet_raw_and_data_sinthetic.csv')
            logging.info(f"Data ingesion completed")
            return df
    
        except Exception as e:
            raise WeightException(e,sys) 
        
    def process_data(self):
        df=self.data_ingestion1()
        def weight_classification(i):
            if i=='Normal_Weight':
                return 'Normal'
            elif i=='Overweight_Level_I' or i=='Overweight_Level_II':
                return 'Overweight'
            elif i=='Insufficient_Weight':
                return 'Underweight'
            elif i=='Obesity_Type_I' or i=='Obesity_Type_II':
                return "Obesity"
            else:
                return 'Extreme Obesity'
        df['NObeyesdad']= df['NObeyesdad'].apply(weight_classification)
        return df
    def split(self):
        try:

            train_set,test_set=train_test_split(self.process_data(),random_state=42,train_size=0.75)
            
            return train_set,test_set
        except Exception as e:
            raise WeightException(e,sys)
        

