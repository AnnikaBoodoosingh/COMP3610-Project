from flask import Flask, render_template, request
import pandas as pd
import xgboost as xgb
import joblib
import numpy as np
from datetime import datetime

app = Flask(__name__)  

# Load assets
carrier_delay_rate = joblib.load('models/carrier_delay_rate.pkl')
origin_delay_rate = joblib.load('models/origin_delay_rate.pkl')
dest_delay_rate = joblib.load('models/dest_delay_rate.pkl')
weather_delay_rate = joblib.load('models/weather_delay_rate.pkl')
nas_delay_rate_by_route = joblib.load('models/nas_delay_rate_by_route.pkl')
late_aircraft_delay_rate = joblib.load('models/late_aircraft_delay_rate.pkl')
flight_delay_sum = joblib.load('models/flight_delay_sum.pkl')
scaler = joblib.load('models/scaler.pkl')

xgb_model = xgb.Booster()
xgb_model.load_model('models/xgb_model.json')

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None

    if request.method == 'POST':
        try:
            # Parse form inputs
            date = request.form['date']
            arr_time = int(request.form['arrTime'])
            crs_elapsed = int(request.form['crsElapsedTime'])
            distance = int(request.form['distance'])
            airline = request.form['airline']
            flight_num = request.form['flightNumber']
            origin = request.form['origin']
            dest = request.form['dest']
            origin_state = request.form['originState']
            dest_state = request.form['destState']

            # Parse date components
            dt = datetime.strptime(date, '%Y-%m-%d')
            year, month, day = dt.year, dt.month, dt.day
            day_of_week = dt.weekday()  

            df = pd.DataFrame([{
                'Year': year,
                'Quarter': (month - 1) // 3 + 1,
                'Month': month,
                'DayofMonth': day,
                'DayOfWeek': day_of_week,
                'CRSElapsedTime': crs_elapsed,
                'Distance': distance,
                'CRSArrTime': arr_time,
                'IATA_CODE_Reporting_Airline': airline,
                'Origin': origin,
                'Dest': dest,
                'OriginState': origin_state,
                'DestState': dest_state,
                'Flight_Number_Reporting_Airline': flight_num
            }])

            # Feature engineering
            df['ArrTime_combined'] = df['CRSArrTime'] // 100 * 60 + df['CRSArrTime'] % 100
            df['DayArrivalFlight'] = df['ArrTime_combined'].apply(lambda x: 1 if 360 <= x <= 840 else 0)
            df['HighDelaySeason'] = df['Month'].apply(lambda x: 1 if x in [1, 3, 6, 7, 12] else 0)
            df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x in [5, 6] else 0)
            df['HighDelayAirline'] = df['IATA_CODE_Reporting_Airline'].apply(lambda x: 1 if x in ['PI', 'AL', 'AA', 'B6', 'PS'] else 0)
            df['HighDelayWeekday'] = df['DayOfWeek'].apply(lambda x: 1 if x in [3, 4] else 0)

            high_delay_origin_states = ['CA', 'TX', 'IL', 'FL', 'GA', 'NY', 'CO', 'NC', 'PA', 'AZ']
            high_delay_dest_states = ['CA', 'TX', 'FL', 'IL', 'GA', 'NY', 'NC', 'CO', 'PA', 'AZ']
            df['HighDelayOriginState'] = df['OriginState'].apply(lambda x: 1 if x in high_delay_origin_states else 0)
            df['HighDelayDestState'] = df['DestState'].apply(lambda x: 1 if x in high_delay_dest_states else 0)

            df['Route'] = df['Origin'] + '-' + df['Dest']

            df['AirlineDelayRate'] = df['IATA_CODE_Reporting_Airline'].map(carrier_delay_rate)
            df['OriginStateDelayRate'] = df['OriginState'].map(origin_delay_rate)
            df['DestStateDelayRate'] = df['DestState'].map(dest_delay_rate)
            df['WeatherDelayRate'] = df['Route'].map(weather_delay_rate)
            df['NASDelayRate'] = df['Route'].map(nas_delay_rate_by_route)
            df['LateAircraftDelayRate'] = df['IATA_CODE_Reporting_Airline'].map(late_aircraft_delay_rate)
            df['HighestDelayedFlightNumber'] = df['Flight_Number_Reporting_Airline'].map(
                flight_delay_sum.set_index('Flight_Number_Reporting_Airline')['DelayCategory']
            )

            df.fillna(0, inplace=True)

            features = ['Year', 'Quarter', 'Month', 'DayofMonth', 'DayOfWeek', 'CRSElapsedTime', 'Distance',
                        'HighDelaySeason', 'DayArrivalFlight', 'IsWeekend', 'HighDelayAirline',
                        'HighDelayOriginState', 'HighDelayDestState', 'HighDelayWeekday', 'AirlineDelayRate',
                        'OriginStateDelayRate', 'DestStateDelayRate', 'WeatherDelayRate',
                        'NASDelayRate', 'LateAircraftDelayRate', 'HighestDelayedFlightNumber']

            X_input_scaled = scaler.transform(df[features])
            dmat = xgb.DMatrix(X_input_scaled)
            prob = xgb_model.predict(dmat)[0]
            pred = "Delayed" if prob >= 0.5 else "On-Time"
            prediction = f"{pred} (probability of delay: {prob:.2f})"

        except Exception as e:
            prediction = f"Error: {e}"

    return render_template('website.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
