from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

# Allow all origins for CORS
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Define the input data model
class ModelInput(BaseModel):
    Time: float
    V1: float
    V2: float
    V3: float
    V4: float
    V5: float
    V6: float
    V7: float
    V8: float
    V9: float
    V10: float
    V11: float
    V12: float
    V13: float
    V14: float
    V15: float
    V16: float
    V17: float
    V18: float
    V19: float
    V20: float
    V21: float
    V22: float
    V23: float
    V24: float
    V25: float
    V26: float
    V27: float
    V28: float
    Amount: float

# Load the saved model
cc_model3 = pickle.load(open('bestcc_model3.sav', 'rb'))

@app.post('/ccmodel_prediction')
def ccmodel_pred(input_parameters: ModelInput):
    # Convert input parameters to JSON, then to a dictionary
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)

    # Extract the values from the dictionary
    input_list = [input_dictionary[feature] for feature in input_dictionary]

    # Make a prediction using the model
    prediction = cc_model3.predict([input_list])

    # Return the prediction result
    if prediction[0] <= 0.8:
        return {"Class": 0}
    else:
        return {"Class": 1}
