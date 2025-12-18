import os, sys
import pandas as pd
import numpy as np
from textblob import TextBlob
from dataclasses import dataclass
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, OrdinalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_selection import SelectKBest, f_classif


from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformartionConfigs:
    preprocess_obj_file_patrh = os.path.join("artifacts/data_transformation", "preprcessor.pkl")
    feature_selector_path = os.path.join("artifacts/data_transformation", "feature_selector.pkl")



class SentimentExtractor(BaseEstimator, TransformerMixin):
    def fit(self, X, y=None):
        return self
    def transform(self, X):
        return X["Reviews"].fillna("").apply(lambda x: TextBlob(str(x)).sentiment.polarity).values.reshape(-1, 1)



class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformartionConfigs()

    def get_data_transformation_obj(self):
        try:
            logging.info("Data Transformation Started")

            numerical_features = ['Age', 'Family size']

            ordinal_features = ['Occupation', 'Monthly Income', 'Educational Qualifications', 'Ease and convenient',
                'Time saving', 'More restaurant choices', 'Easy Payment option', 'More Offers and Discount', 
                'Good Food quality', 'Good Tracking system', 'Self Cooking', 'Health Concern', 'Late Delivery',
                'Poor Hygiene', 'Bad past experience', 'Unavailability', 'Unaffordable', 'Long delivery time',
                'Delay of delivery person getting assigned', 'Delay of delivery person picking up food', 
                'Wrong order delivered', 'Missing item', 'Order placed by mistake', 'Influence of time',
                'Maximum wait time', 'Residence in busy location', 'Google Maps Accuracy', 'Good Road Condition', 
                'Low quantity low time', 'Delivery person ability', 'Influence of rating', 'Less Delivery time',
                'High Quality of package', 'Number of calls', 'Politeness', 'Freshness ', 'Temperature', 'Good Taste ',
                'Good Quantity']

            ordinal_categories = [
                ['Student', 'House wife', 'Self Employeed', 'Employee'],
                ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'],
                ['Uneducated', 'School', 'Graduate', 'Post Graduate', 'Ph.D'],

                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'], 
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'], 
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'], 
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'], 
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'], 
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'], 
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'], 
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'], 
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'], 
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'], 
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  


                ['No', 'Maybe', 'Yes'],


                ['15 minutes', '30 minutes', '45 minutes', '60 minutes', 'More than 60 minutes'],


                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],


                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'], 
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],  


                ['No', 'Maybe', 'Yes'],


                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'], 
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'], 
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'], 
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'], 
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'], 
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'], 
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'], 
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'] 
            ]


            nominal_features = ['Gender', 'Marital Status', 'Order Time', 'Perference(P1)', 'Perference(P2)']

            ordinal_encoder = OrdinalEncoder(categories=ordinal_categories)

            num_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='median')),
                ("scaler", StandardScaler())
            ])

            ord_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("encoder", ordinal_encoder),
                ("scaler", StandardScaler())
            ])

            nom_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy='most_frequent')),
                ("encoder", OneHotEncoder(handle_unknown='ignore'))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_features),
                ("ord_pipeline", ord_pipeline, ordinal_features),
                ("nom_pipeline", nom_pipeline, nominal_features),
                ("sentiment", SentimentExtractor(), ["Reviews"])
            ])

            selector = SelectKBest(score_func=f_classif, k=20)
            return preprocessor, selector


        except Exception as e:
            raise CustomException(e, sys)

    def remove_outliers_IQR(self, col, df):
        try:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            iqr = Q3 - Q1
            upper_limit = Q3 + 1.5 * iqr
            lower_limit = Q1 - 1.5 * iqr
            df.loc[df[col] > upper_limit, col] = upper_limit
            df.loc[df[col] < lower_limit, col] = lower_limit
            return df

        except Exception as e:
            logging.info("Outliers handling code")
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_data = pd.read_csv(train_path)
            test_data = pd.read_csv(test_path)


            numerical_features = ['Age', 'Family size']

            for col in numerical_features:
                self.remove_outliers_IQR(col=col, df=train_data)
            logging.info("Outliers capped on train data")

            for col in numerical_features:
                self.remove_outliers_IQR(col=col, df=test_data)
            logging.info("Outliers capped on test data")

            preprocess_obj, selector = self.get_data_transformation_obj()


            target_column = "Output"
            drop_columns = [target_column]

            logging.info("Splitting train data")
            input_feature_train_data = train_data.drop(drop_columns, axis=1)
            target_feature_train_data = train_data[target_column]

            logging.info("Splitting test data")
            input_feature_test_data = test_data.drop(drop_columns, axis=1)
            target_feature_test_data = test_data[target_column]

            X_train_transformed = preprocess_obj.fit_transform(input_feature_train_data)
            X_test_transformed = preprocess_obj.transform(input_feature_test_data)
            
            X_train_selected = selector.fit_transform(X_train_transformed, target_feature_train_data)
            X_test_selected = selector.transform(X_test_transformed)

            
            train_array = np.c_[X_train_selected, target_feature_train_data]
            test_array = np.c_[X_test_selected, target_feature_test_data]


            save_object(
            file_path=self.data_transformation_config.feature_selector_path,
            obj=selector
        )


            return (
            train_array,
            test_array,
            self.data_transformation_config.preprocess_obj_file_patrh,
            self.data_transformation_config.feature_selector_path
        )


        except Exception as e:
            raise CustomException(e, sys)
