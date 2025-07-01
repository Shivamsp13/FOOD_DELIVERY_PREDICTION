import os, sys
from src.logger import logging
from src.exception import CustomException
import numpy as np
import pandas as pd
from dataclasses import dataclass
from src.utils import load_object

class PredictionPipeline:
    def __init__(self):
        pass
    
    def predict(self, features):
        try:
            preprocessor_path = os.path.join("artifacts/data_transformation", "preprcessor.pkl")
            model_path = os.path.join("artifacts/model_trainer", "model.pkl")
            
            logging.info("Loading preprocessor and model")
            processor = load_object(preprocessor_path)
            model = load_object(model_path)
            
            logging.info("Transforming input features")
            scaled = processor.transform(features)
            
            logging.info("Making prediction")
            pred = model.predict(scaled)
            
            return pred
            
        except Exception as e:
            raise CustomException(e, sys)

class CustomClass:
    def __init__(self,
                 age: int,
                 family_size: int,
                 gender: str,
                 marital_status: str,
                 occupation: str,
                 monthly_income: str,
                 educational_qualifications: str,
                 order_time: str,
                 ease_convenient: str,
                 time_saving: str,
                 more_restaurant_choices: str,
                 easy_payment_option: str,
                 more_offers_discount: str,
                 good_food_quality: str,
                 good_tracking_system: str,
                 self_cooking: str,
                 health_concern: str,
                 late_delivery: str,
                 poor_hygiene: str,
                 bad_past_experience: str,
                 unavailability: str,
                 unaffordable: str,
                 long_delivery_time: str,
                 delay_delivery_person_assigned: str,
                 delay_delivery_person_pickup: str,
                 wrong_order_delivered: str,
                 missing_item: str,
                 order_placed_mistake: str,
                 influence_time: str,
                 maximum_wait_time: str,
                 residence_busy_location: str,
                 google_maps_accuracy: str,
                 good_road_condition: str,
                 low_quantity_low_time: str,
                 delivery_person_ability: str,
                 influence_rating: str,
                 less_delivery_time: str,
                 high_quality_package: str,
                 number_of_calls: str,
                 politeness: str,
                 freshness: str,
                 temperature: str,
                 good_taste: str,
                 good_quantity: str,
                 preference_p1: str,
                 preference_p2: str,
                 reviews: str = ""):
        
        # Numerical features
        self.age = age
        self.family_size = family_size
        
        # Nominal features
        self.gender = gender
        self.marital_status = marital_status
        self.order_time = order_time
        self.preference_p1 = preference_p1
        self.preference_p2 = preference_p2
        
        # Ordinal features
        self.occupation = occupation
        self.monthly_income = monthly_income
        self.educational_qualifications = educational_qualifications
        
        # Likert scale features (ordinal)
        self.ease_convenient = ease_convenient
        self.time_saving = time_saving
        self.more_restaurant_choices = more_restaurant_choices
        self.easy_payment_option = easy_payment_option
        self.more_offers_discount = more_offers_discount
        self.good_food_quality = good_food_quality
        self.good_tracking_system = good_tracking_system
        self.self_cooking = self_cooking
        self.health_concern = health_concern
        self.late_delivery = late_delivery
        self.poor_hygiene = poor_hygiene
        self.bad_past_experience = bad_past_experience
        self.unavailability = unavailability
        self.unaffordable = unaffordable
        self.long_delivery_time = long_delivery_time
        self.delay_delivery_person_assigned = delay_delivery_person_assigned
        self.delay_delivery_person_pickup = delay_delivery_person_pickup
        self.wrong_order_delivered = wrong_order_delivered
        self.missing_item = missing_item
        self.order_placed_mistake = order_placed_mistake
        self.residence_busy_location = residence_busy_location
        self.google_maps_accuracy = google_maps_accuracy
        self.good_road_condition = good_road_condition
        self.low_quantity_low_time = low_quantity_low_time
        self.delivery_person_ability = delivery_person_ability
        
        # Yes/No/Maybe features
        self.influence_time = influence_time
        self.influence_rating = influence_rating
        
        # Wait time feature
        self.maximum_wait_time = maximum_wait_time
        
        # Importance scale features
        self.less_delivery_time = less_delivery_time
        self.high_quality_package = high_quality_package
        self.number_of_calls = number_of_calls
        self.politeness = politeness
        self.freshness = freshness
        self.temperature = temperature
        self.good_taste = good_taste
        self.good_quantity = good_quantity
        
        # Text feature for sentiment analysis
        self.reviews = reviews
    
    def get_data_DataFrame(self):
        try:
            custom_input = {
                # Numerical features
                "Age": [self.age],
                "Family size": [self.family_size],
                
                # Nominal features
                "Gender": [self.gender],
                "Marital Status": [self.marital_status],
                "Order Time": [self.order_time],
                "Perference(P1)": [self.preference_p1],
                "Perference(P2)": [self.preference_p2],
                
                # Ordinal features
                "Occupation": [self.occupation],
                "Monthly Income": [self.monthly_income],
                "Educational Qualifications": [self.educational_qualifications],
                
                # Likert scale features
                "Ease and convenient": [self.ease_convenient],
                "Time saving": [self.time_saving],
                "More restaurant choices": [self.more_restaurant_choices],
                "Easy Payment option": [self.easy_payment_option],
                "More Offers and Discount": [self.more_offers_discount],
                "Good Food quality": [self.good_food_quality],
                "Good Tracking system": [self.good_tracking_system],
                "Self Cooking": [self.self_cooking],
                "Health Concern": [self.health_concern],
                "Late Delivery": [self.late_delivery],
                "Poor Hygiene": [self.poor_hygiene],
                "Bad past experience": [self.bad_past_experience],
                "Unavailability": [self.unavailability],
                "Unaffordable": [self.unaffordable],
                "Long delivery time": [self.long_delivery_time],
                "Delay of delivery person getting assigned": [self.delay_delivery_person_assigned],
                "Delay of delivery person picking up food": [self.delay_delivery_person_pickup],
                "Wrong order delivered": [self.wrong_order_delivered],
                "Missing item": [self.missing_item],
                "Order placed by mistake": [self.order_placed_mistake],
                "Residence in busy location": [self.residence_busy_location],
                "Google Maps Accuracy": [self.google_maps_accuracy],
                "Good Road Condition": [self.good_road_condition],
                "Low quantity low time": [self.low_quantity_low_time],
                "Delivery person ability": [self.delivery_person_ability],
                
                # Yes/No/Maybe features
                "Influence of time": [self.influence_time],
                "Influence of rating": [self.influence_rating],
                
                # Wait time feature
                "Maximum wait time": [self.maximum_wait_time],
                
                # Importance scale features
                "Less Delivery time": [self.less_delivery_time],
                "High Quality of package": [self.high_quality_package],
                "Number of calls": [self.number_of_calls],
                "Politeness": [self.politeness],
                "Freshness ": [self.freshness],
                "Temperature": [self.temperature],
                "Good Taste ": [self.good_taste],
                "Good Quantity": [self.good_quantity],
                
                # Text feature for sentiment analysis
                "Reviews": [self.reviews]
            }
            
            data = pd.DataFrame(custom_input)
            logging.info("DataFrame created successfully")
            return data
            
        except Exception as e:
            raise CustomException(e, sys)