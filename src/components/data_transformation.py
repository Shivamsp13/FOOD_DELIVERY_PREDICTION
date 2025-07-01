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

from src.logger import logging
from src.exception import CustomException
from src.utils import save_object

@dataclass
class DataTransformartionConfigs:
    preprocess_obj_file_patrh = os.path.join("artifacts/data_transformation", "preprcessor.pkl")


# Sentiment extractor for review column
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
                # 1–3: Single category per feature
                ['Student', 'House wife', 'Self Employeed', 'Employee'],  # Occupation
                ['No Income', 'Below Rs.10000', '10001 to 25000', '25001 to 50000', 'More than 50000'],  # Monthly Income
                ['Uneducated', 'School', 'Graduate', 'Post Graduate', 'Ph.D'],  # Educational Qualifications

                # 4–24: 21 Likert-scale features
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Ease and convenient
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Time saving
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # More restaurant choices
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Easy Payment option
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # More Offers and Discount
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Good Food quality
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Good Tracking system
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Self Cooking
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Health Concern
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Late Delivery
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Poor Hygiene
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Bad past experience
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Unavailability
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Unaffordable
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Long delivery time
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Delay of delivery person getting assigned
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Delay of delivery person picking up food
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Wrong order delivered
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Missing item
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly agree'],  # Order placed by mistake

                # 25: Influence of time
                ['No', 'Maybe', 'Yes'],

                # 26: Maximum wait time
                ['15 minutes', '30 minutes', '45 minutes', '60 minutes', 'More than 60 minutes'],

                # 27: Residence in busy location — THIS was missing before
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],

                # 28–31: Next 4 Likert-scale fields
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],  # Google Maps Accuracy
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],  # Good Road Condition
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],  # Low quantity low time
                ['Strongly disagree', 'Disagree', 'Neutral', 'Agree', 'Strongly Agree'],  # Delivery person ability

                # 32: Influence of rating
                ['No', 'Maybe', 'Yes'],

                # 33–40: 8 Importance-based features
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'],  # Less Delivery time
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'],  # High Quality of package
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'],  # Number of calls
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'],  # Politeness
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'],  # Freshness 
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'],  # Temperature
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important'],  # Good Taste 
                ['Unimportant', 'Slightly Important', 'Moderately Important', 'Important', 'Very Important']  # Good Quantity
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

            return preprocessor

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

            # Example columns (update these to match actual numeric features for outlier handling)
            numerical_features = ['Age', 'Family size']

            for col in numerical_features:
                self.remove_outliers_IQR(col=col, df=train_data)
            logging.info("Outliers capped on train data")

            for col in numerical_features:
                self.remove_outliers_IQR(col=col, df=test_data)
            logging.info("Outliers capped on test data")

            preprocess_obj = self.get_data_transformation_obj()

            target_column = "Output"
            drop_columns = [target_column]

            logging.info("Splitting train data")
            input_feature_train_data = train_data.drop(drop_columns, axis=1)
            target_feature_train_data = train_data[target_column]

            logging.info("Splitting test data")
            input_feature_test_data = test_data.drop(drop_columns, axis=1)
            target_feature_test_data = test_data[target_column]

            input_train_arr = preprocess_obj.fit_transform(input_feature_train_data)
            input_test_arr = preprocess_obj.transform(input_feature_test_data)

            train_array = np.c_[input_train_arr, np.array(target_feature_train_data)]
            test_array = np.c_[input_test_arr, np.array(target_feature_test_data)]

            save_object(file_path=self.data_transformation_config.preprocess_obj_file_patrh,
                        obj=preprocess_obj)

            return train_array, test_array, self.data_transformation_config.preprocess_obj_file_patrh

        except Exception as e:
            raise CustomException(e, sys)
