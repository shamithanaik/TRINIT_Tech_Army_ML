import numpy as np
from flask import Flask, request, jsonify, render_template
import pickle
from sklearn.metrics import accuracy_score


flask_app = Flask(__name__)

def text_splitter(text):
    return text.split()
with open("model.pkl", "rb") as f:
    model, X_test, y_test = pickle.load(f)
with open("tfidf.pkl","rb") as g:
    tfidf=pickle.load(g)
with open("model_1.pkl","rb") as i:
    model_1=pickle.load(i)
with open("model_2.pkl","rb") as j:
    model_2=pickle.load(j)
with open("model_3.pkl","rb") as k:
    model_3=pickle.load(k)


@flask_app.route("/")
def Home():
    return render_template("index.html")

@flask_app.route("/predict", methods=["POST"])
def predict():
    if request.method == 'POST':
        comment = request.form['Comment']
        list_a=[]
        list_a.append(comment)
        val_comment=tfidf.transform(list_a)
        prediction_text = model.predict(val_comment)
        str=""
        if prediction_text[0][0]==1:
            str+="Commenting "
        if prediction_text[0][1]==1:
            str+="Facial Expression "
        if prediction_text[0][2]==1:
            str+="Grouping "
        prediction_text=str
            
        return render_template('index.html',prediction_text=prediction_text)
@flask_app.route("/predict_1", methods=["POST"])
def predict_1():
    
    selected_option = request.form['option']
    comment=request.form['Comment_1']
    list_a=[]
    list_a.append(comment)
    val_comment=tfidf.transform(list_a)
    
    if selected_option=="option1":
        y_pred=model_1.predict(val_comment)
        str=""
        if y_pred[0]==0:
            str="Non_sexual"

        else:
            str="sexual"
        return f'The description given for sexual assault is: {str}'
    elif selected_option=="option2":
        y_pred=model_2.predict(val_comment)
        str=""
        if y_pred[0]==0:
            str="Non_sexual"

        else:
            str="sexual"
        return f'The description given for sexual assault is: {str}'
    elif selected_option=="option3":
        y_pred=model_3.predict(val_comment)
        str=""
        if y_pred[0]==0:
            str="Non_sexual"

        else:
            str="sexual"
        return f'The description given for sexual assault is: {str}'
    
   
    
    
   
    

if __name__ == "__main__":
    flask_app.run(debug=True)
