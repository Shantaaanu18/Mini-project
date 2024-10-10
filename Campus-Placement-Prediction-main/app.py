import numpy as np
from flask import Flask, request, render_template
import pickle
import warnings
warnings.simplefilter("ignore", UserWarning)

# Create flask app
app = Flask(__name__)
model = pickle.load(open("model.pkl", "rb"))
def ValuePredictor(to_predict_list):
    to_predict = np.array(to_predict_list).reshape(1, 12)
    loaded_model = pickle.load(open("model.pkl", "rb"))
    result = loaded_model.predict(to_predict)
    probability = loaded_model.predict_proba(to_predict)[:, 1]  # Probability of positive class (placed)
    return int(result[0]), probability[0]

# prediction function
# def ValuePredictor(to_predict_list):
# 	to_predict = np.array(to_predict_list).reshape(1, 12)
# 	loaded_model = pickle.load(open("model.pkl", "rb"))
# 	result = loaded_model.predict(to_predict)
# 	return result[0]

# def calculate_accuracy(X_test, y_test):
#     loaded_model = pickle.load(open("model.pkl", "rb"))
#     y_pred = loaded_model.predict(X_test)
#     accuracy = np.mean(y_pred == y_test)
#     return accuracy

@app.route("/")
def Home():
    print('Request for index page received')
    return render_template("index.html")

@app.route("/result", methods = ["POST"])
def result():
    print('Request for predict page received')
    if request.method == 'POST':
        to_predict_list = request.form.to_dict()
        to_predict_list = list(to_predict_list.values())
        to_predict_list = list(map(int, to_predict_list))
        result, probability = ValuePredictor(to_predict_list)
        if int(result)== 1:
            prediction ='Placed'
        else:
            prediction ='Not Placed'
        X_test = np.array(to_predict_list).reshape(1, 12)  # Assuming X_test is available
        y_test = np.array([1])  # Assuming y_test is available
        # accuracy = calculate_accuracy(X_test, y_test)
        # a=accuracy*100
        # print("Accuracy:", accuracy)
    return render_template("result.html", prediction_text=prediction, probability_text=(probability*100))

if __name__ == "__main__":
    app.run(debug=True)