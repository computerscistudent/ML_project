import os
import sys
from src.exception import customException
from src.logger import logging
import pandas as pd
from sklearn.model_selection import train_test_split
from dataclasses import dataclass
from src.components.data_transformation import DataTransformation
from src.components.data_transformation import DataFormation



@dataclass
class DataIngestionConfig :
    train_data_path = os.path.join('artifact', 'train_data.csv') 
    test_data_path = os.path.join('artifact', 'test_data.csv')
    raw_data_path = os.path.join('artifact', 'raw_data.csv')


class DataIngestion:
    def __init__(self):
        self.ingestion_config = DataIngestionConfig()

    def initialize_data_ingestion(self):
        logging.info("Entered the data Ingestion Method or Component")
        try:
            df = pd.read_csv(r"notebook\data\stud.csv")
            logging.info("read the dataset as DataFrame")

            os.makedirs(os.path.dirname(self.ingestion_config.train_data_path),exist_ok=True)
            df.to_csv(self.ingestion_config.raw_data_path,index=False,header=True)

            logging.info("train_test_split has been initialized")
            train_data,test_data = train_test_split(df,test_size=0.3,random_state=42)
            train_data.to_csv(self.ingestion_config.train_data_path,index=False,header=True)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False,header=True)

            logging.info("Ingestion of the data is completed")

            return (
                self.ingestion_config.test_data_path,
                self.ingestion_config.train_data_path
            )
        except Exception as e:
            raise customException(e)   
        
if __name__ == "__main__":
    obj = DataIngestion()
    test_data,train_data = obj.initialize_data_ingestion()      

    processed_data = DataTransformation()
    processed_data.initiate_data_transformation(test_data,train_data)  

        
