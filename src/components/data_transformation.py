from sklearn.preprocessing import OneHotEncoder,StandardScaler
from src.exception import customException
from src.logger import logging
import os
from sklearn.pipeline import Pipeline
from dataclasses import dataclass
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
import pandas as pd

@dataclass
class DataFormation:
    preprocessor_obj_file_path = os.path.join('artifact', 'preprocessor.pkl')

class DataTransformation:
    
    def __init__(self):
            self.data_tranformation_config = DataFormation()

    def get_data_transformer_object(self):
        try:
            numerical_columns = ["reading_score","writing_score"]
            catagorical_columns = ["gender",
                                    "race_ethnicity",
                                    "parental_level_of_education",
                                    "lunch",
                                    "test_preparation_course"]
                    
            num_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="median")),
                    ("scaler",StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps = [
                    ("imputer",SimpleImputer(strategy="mode")),
                    ("encoder",OneHotEncoder()),
                    ("scaler",StandardScaler())
                ]
            )
            logging.info(f"Categorical Columns: {catagorical_columns}")
            logging.info(f"Numerical Columns: {numerical_columns}")

            preprocessor = ColumnTransformer(
                [
                    ("num_pipeline",num_pipeline,numerical_columns),
                    ("cat_pipeline",cat_pipeline,catagorical_columns)
                ]
            )
            logging.info("preprocessing has been completed")
            return preprocessor
        except:
            pass
            
    def initiate_data_transformation(self, train_path, test_path):
        try:
            
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data completed")

            
            logging.info("Obtaining preprocessing object")
            preprocessing_obj = self.get_data_transformer_object()

            
            
        except Exception as e:
            raise customException(e)
       
    



