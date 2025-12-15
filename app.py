from flask import Flask, request, render_template, jsonify
import pandas as pd
import numpy as np
from src.pipeline.prediction_pipeline import CustomClass, PredictionPipeline
from src.exception import CustomException
from src.logger import logging
import sys

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict_datapoint', methods=['GET', 'POST'])
def predict_datapoint():
    if request.method == 'GET':
        return render_template('home.html')
    else:
        try:
            data = CustomClass(
                age=int(request.form.get('age')),
                family_size=int(request.form.get('family_size')),
                gender=request.form.get('gender'),
                marital_status=request.form.get('marital_status'),
                occupation=request.form.get('occupation'),
                monthly_income=request.form.get('monthly_income'),
                educational_qualifications=request.form.get('educational_qualifications'),
                order_time=request.form.get('order_time'),
                ease_convenient=request.form.get('ease_convenient'),
                time_saving=request.form.get('time_saving'),
                more_restaurant_choices=request.form.get('more_restaurant_choices'),
                easy_payment_option=request.form.get('easy_payment_option'),
                more_offers_discount=request.form.get('more_offers_discount'),
                good_food_quality=request.form.get('good_food_quality'),
                good_tracking_system=request.form.get('good_tracking_system'),
                self_cooking=request.form.get('self_cooking'),
                health_concern=request.form.get('health_concern'),
                late_delivery=request.form.get('late_delivery'),
                poor_hygiene=request.form.get('poor_hygiene'),
                bad_past_experience=request.form.get('bad_past_experience'),
                unavailability=request.form.get('unavailability'),
                unaffordable=request.form.get('unaffordable'),
                long_delivery_time=request.form.get('long_delivery_time'),
                delay_delivery_person_assigned=request.form.get('delay_delivery_person_assigned'),
                delay_delivery_person_pickup=request.form.get('delay_delivery_person_pickup'),
                wrong_order_delivered=request.form.get('wrong_order_delivered'),
                missing_item=request.form.get('missing_item'),
                order_placed_mistake=request.form.get('order_placed_mistake'),
                influence_time=request.form.get('influence_time'),
                maximum_wait_time=request.form.get('maximum_wait_time'),
                residence_busy_location=request.form.get('residence_busy_location'),
                google_maps_accuracy=request.form.get('google_maps_accuracy'),
                good_road_condition=request.form.get('good_road_condition'),
                low_quantity_low_time=request.form.get('low_quantity_low_time'),
                delivery_person_ability=request.form.get('delivery_person_ability'),
                influence_rating=request.form.get('influence_rating'),
                less_delivery_time=request.form.get('less_delivery_time'),
                high_quality_package=request.form.get('high_quality_package'),
                number_of_calls=request.form.get('number_of_calls'),
                politeness=request.form.get('politeness'),
                freshness=request.form.get('freshness'),
                temperature=request.form.get('temperature'),
                good_taste=request.form.get('good_taste'),
                good_quantity=request.form.get('good_quantity'),
                preference_p1=request.form.get('preference_p1'),
                preference_p2=request.form.get('preference_p2'),
                reviews=request.form.get('reviews', '')
            )
            
            pred_df = data.get_data_DataFrame()
            logging.info("DataFrame created from form data")
            
            predict_pipeline = PredictionPipeline()
            results = predict_pipeline.predict(pred_df)
            
            prediction_text = "Will Use Food Delivery" if results[0] == 1 else "Will Not Use Food Delivery"
            
            logging.info(f"Prediction completed: {prediction_text}")
            return render_template('home.html', results=prediction_text)
            
        except Exception as e:
            logging.error(f"Error in prediction: {str(e)}")
            raise CustomException(e, sys)

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for programmatic predictions"""
    try:
        json_data = request.get_json()
        
        required_fields = ['age', 'family_size', 'gender', 'marital_status', 'occupation']
        for field in required_fields:
            if field not in json_data:
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        data = CustomClass(
            age=json_data.get('age'),
            family_size=json_data.get('family_size'),
            gender=json_data.get('gender'),
            marital_status=json_data.get('marital_status'),
            occupation=json_data.get('occupation'),
            monthly_income=json_data.get('monthly_income', 'Below Rs.10000'),
            educational_qualifications=json_data.get('educational_qualifications', 'Graduate'),
            order_time=json_data.get('order_time', 'Lunch'),
            ease_convenient=json_data.get('ease_convenient', 'Neutral'),
            time_saving=json_data.get('time_saving', 'Neutral'),
            more_restaurant_choices=json_data.get('more_restaurant_choices', 'Neutral'),
            easy_payment_option=json_data.get('easy_payment_option', 'Neutral'),
            more_offers_discount=json_data.get('more_offers_discount', 'Neutral'),
            good_food_quality=json_data.get('good_food_quality', 'Neutral'),
            good_tracking_system=json_data.get('good_tracking_system', 'Neutral'),
            self_cooking=json_data.get('self_cooking', 'Neutral'),
            health_concern=json_data.get('health_concern', 'Neutral'),
            late_delivery=json_data.get('late_delivery', 'Neutral'),
            poor_hygiene=json_data.get('poor_hygiene', 'Neutral'),
            bad_past_experience=json_data.get('bad_past_experience', 'Neutral'),
            unavailability=json_data.get('unavailability', 'Neutral'),
            unaffordable=json_data.get('unaffordable', 'Neutral'),
            long_delivery_time=json_data.get('long_delivery_time', 'Neutral'),
            delay_delivery_person_assigned=json_data.get('delay_delivery_person_assigned', 'Neutral'),
            delay_delivery_person_pickup=json_data.get('delay_delivery_person_pickup', 'Neutral'),
            wrong_order_delivered=json_data.get('wrong_order_delivered', 'Neutral'),
            missing_item=json_data.get('missing_item', 'Neutral'),
            order_placed_mistake=json_data.get('order_placed_mistake', 'Neutral'),
            influence_time=json_data.get('influence_time', 'Maybe'),
            maximum_wait_time=json_data.get('maximum_wait_time', '30 minutes'),
            residence_busy_location=json_data.get('residence_busy_location', 'Neutral'),
            google_maps_accuracy=json_data.get('google_maps_accuracy', 'Neutral'),
            good_road_condition=json_data.get('good_road_condition', 'Neutral'),
            low_quantity_low_time=json_data.get('low_quantity_low_time', 'Neutral'),
            delivery_person_ability=json_data.get('delivery_person_ability', 'Neutral'),
            influence_rating=json_data.get('influence_rating', 'Maybe'),
            less_delivery_time=json_data.get('less_delivery_time', 'Moderately Important'),
            high_quality_package=json_data.get('high_quality_package', 'Moderately Important'),
            number_of_calls=json_data.get('number_of_calls', 'Moderately Important'),
            politeness=json_data.get('politeness', 'Moderately Important'),
            freshness=json_data.get('freshness', 'Moderately Important'),
            temperature=json_data.get('temperature', 'Moderately Important'),
            good_taste=json_data.get('good_taste', 'Moderately Important'),
            good_quantity=json_data.get('good_quantity', 'Moderately Important'),
            preference_p1=json_data.get('preference_p1', 'Ease and convenient'),
            preference_p2=json_data.get('preference_p2', 'Time saving'),
            reviews=json_data.get('reviews', '')
        )
        
        pred_df = data.get_data_DataFrame()
        predict_pipeline = PredictionPipeline()
        results = predict_pipeline.predict(pred_df)
        
        prediction_result = {
            'prediction': int(results[0]),
            'prediction_text': "Will Use Food Delivery" if results[0] == 1 else "Will Not Use Food Delivery",
            'probability': float(results[0]) if hasattr(results[0], 'item') else float(results[0])
        }
        
        return jsonify(prediction_result)
        
    except Exception as e:
        logging.error(f"API Error: {str(e)}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health_check():
    """Health check endpoint"""
    return jsonify({'status': 'healthy', 'message': 'Food Delivery Prediction API is running'})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
