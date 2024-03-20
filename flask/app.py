# import pickle
# import pandas as pd
# import numpy as np

# # Load the pickle file
# with open('weather_pred_dt.pkl', 'rb') as file:
#     model = pickle.load(file)

# # Hardcoded input values for prediction
# wind_deg = 180.0
# pressure = 1015.0
# humidity = 70.0
# temp_max = 25.0
# temp_min = 15.0

# # Make a DataFrame of the input values
# df=pd.DataFrame([wind_deg, pressure, humidity, temp_max, temp_min])
# df=df.T

# # Predict using the model
# prediction = model.predict(df)
# # Round the prediction to 2 decimal places
# prediction = np.round(prediction, 2)

# print("Predicted weather condition:", prediction)


import warnings
warnings.filterwarnings("ignore", category=UserWarning)

from flask import Flask, render_template, request
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing import power_transform


app = Flask(__name__)

# Load the pickle file 
with open('weather_pred_dt.pkl', 'rb') as file:
    model = pickle.load(file)     

@app.route('/predict', methods=['POST'])
def predict():
    # Get the input values from the form
    wind_deg = float(request.form['wind_deg'])
    pressure = float(request.form['pressure'])
    humidity = float(request.form['humidity'])
    temp_max = float(request.form['temp_max'])
    temp_min = float(request.form['temp_min'])
    #make df of above values
    df=pd.DataFrame([wind_deg, pressure, humidity, temp_max, temp_min])
    df=df.T

    print(df)
 
    prediction = model.predict(df)
   
   
    print ("Decision Tree ",prediction)	
    # return render_template('result.html', prediction=prediction)
if __name__ == '__main__':
    app.run()



