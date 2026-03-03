import os 
import joblib
from score import score
import numpy as np 

from flask import Flask, request, jsonify

app=Flask(__name__)

### Load the model and Vectorizer 
MODEL_PATH=os.path.join("SAVED_MODELS","best_spam_classifier_model.pkl")
VECTORIZER_PATH=os.path.join("SAVED_MODELS","tfidf_vectorizer.pkl")

model=joblib.load(MODEL_PATH)
vectorizer=joblib.load(VECTORIZER_PATH)


def compute_socre(text,threshold=0.5):
    features=vectorizer.transform([text])   

    if hasattr(model, "predict_proba"):
        propensity=model.predict_proba(features)[0][1]
    else:
        decision_score=model.decision_function(features)[0]
        propensity=1/(1+np.exp(-decision_score))  ### convert to probability using sigmoid function
    prediction=propensity>=threshold
    return bool(prediction), float(propensity)

@app.route("/score",methods=["POST"])

def score_endpoint():
    data=request.get_json()

    text=data.get("text", "")
    threshold=data.get("threshold", 0.5)

    prediction, propensity=compute_socre(text,threshold)

    return jsonify({
        "prediction": prediction,
        "propensity": propensity
    })

if __name__=="__main__":
    app.run(host="127.0.0.1",port=5000,debug=False)