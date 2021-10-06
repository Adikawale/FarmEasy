from flask import Flask,request,render_template
import pickle
import numpy as np

app = Flask(__name__)

model = pickle.load(open('model.pkl','rb'))

@app.route('/')
def home():
    return render_template('start.html')

@app.route('/crop', methods=['POST',"GET"])
def crop():
    return render_template('crop.html')

@app.route('/info', methods=['POST',"GET"])
def info():
    return render_template('info.html')

@app.route('/predict',methods=['POST',"GET"])
def predict():
    int_features=[int(float(x)) for x in request.form.values()]
 
    final = [np.array(list(int_features))]
    print(final)
    prediction = model.predict(final)
    return render_template('crop.html',pred='"{}"'.format(prediction[0]))
    
if __name__ ==  '__main__':
    app.run(debug=True)
