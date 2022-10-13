import pickle
from flask import Flask,request,app,jsonify,url_for,render_template
import numpy as np
import pandas as pd

app=Flask(__name__)

# Loading a pickle file
model=pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    #return 'Hello World'
    return render_template('home.html')

# Creating an API for single input prediction to test on POSTMAN
@app.route('/predict_api',methods=['POST'])
def predict_api():
    data=request.json['data']
    print(data)
    new_data=[list(data.values())]
    output=model.predict(new_data)[0]
    return jsonify(output)

# Creating an API for single input prediction to test on Browser
@app.route('/predict',methods=['POST'])
def predict():
    data=[float(x) for x in request.form.values()]
    final_features = [np.array(data)]
    print(data)
    output=model.predict(final_features)[0]
    print(output)
    #output = round(prediction[0], 2)
    return render_template('home.html', prediction_text="Airfoil pressure is  {}".format(output))

if __name__=="__main__":
    app.run(debug=True)


