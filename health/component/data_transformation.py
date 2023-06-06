from health.exception import WeightException
from health.logger import logging
import pandas as pd
import numpy as np
import sys
from health.component.data_ingestion import data_ingestion
from sklearn.preprocessing import OrdinalEncoder

class data_transformation:
    def __init__(self):
        self.train,self.test=data_ingestion().split()


    def encoding(self):
        try:

            numerical_variable=[i for i in self.train.columns if self.train[i].dtypes!='O']
            categorical_variable=[i for i in self.train.columns if self.train[i].dtypes=='O']
            encoder=OrdinalEncoder()
            train_array=encoder.fit_transform(self.train[categorical_variable])
            test_array=encoder.transform(self.test[categorical_variable])
            train_array_encoded=np.concatenate((np.array(self.train[numerical_variable]),train_array),axis=1)
            test_array_encoded=np.concatenate((np.array(self.test[numerical_variable]),test_array),axis=1)
            logging.info('Data encoded')
            
            return train_array,test_array
        except Exception as e:
            raise WeightException(e,sys)



                                         
                                   
