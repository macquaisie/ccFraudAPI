from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import json
from fastapi.middleware.cors import CORSMiddleware


app = FastAPI()
origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)



class model_input(BaseModel):
    Time : float
    V1 : float
    V2 : float
    V3 : float
    V4 : float
    V5 : float
    V6 : float
    V7 : float
    V8 : float
    V9 : float
    V10 : float
    V11 : float
    V12 : float
    V13 : float
    V14 : float
    V15 : float
    V16 : float
    V17 : float
    V18 : float
    V19 : float
    V20 : float
    V21 : float
    V22 : float
    V23 : float
    V24 : float
    V25 : float
    V26 : float
    V27 : float
    V28 : float
    Amount : float
          
        
# loading the saved model
cc_model3 = pickle.load(open('bestcc_model3.sav', 'rb'))

@app.post('/ccmodel_prediction')
def ccmodel_pred(input_parameters : model_input):
    
    input_data = input_parameters.json()
    input_dictionary = json.loads(input_data)
    
    time = input_dictionary['Time']
    v1 = input_dictionary['V1']
    v2 = input_dictionary['V2']
    v3 = input_dictionary['V3']
    v4 = input_dictionary['V4']
    v5 = input_dictionary['V5']
    v6 = input_dictionary['V6']
    v7 = input_dictionary['V7']
    v8 = input_dictionary['V8']
    v9 = input_dictionary['V9']
    v10 = input_dictionary['V10']
    v11 = input_dictionary['V11']
    v12 = input_dictionary['V12']
    v13 = input_dictionary['V13']
    v14 = input_dictionary['V14']
    v15 = input_dictionary['V15']
    v16 = input_dictionary['V16']
    v17 = input_dictionary['V17']
    v18 = input_dictionary['V18']
    v19 = input_dictionary['V19']
    v20 = input_dictionary['V20']
    v21 = input_dictionary['V21']
    v22 = input_dictionary['V22']
    v23 = input_dictionary['V23']
    v24 = input_dictionary['V24']
    v25 = input_dictionary['V25']
    v26 = input_dictionary['V26']
    v27 = input_dictionary['V27']
    v28 = input_dictionary['V28']
    amount = input_dictionary['Amount']
    
    
    
    
    
    input_list = [time, v1, v2, v3, v4, v5, v6, v7, v8, v9, v10, v11, v12, v13, v14, v15, v16, v17, v18, v19, v20, v21, v22, v23, v24, v25, v26, v27, v28, amount]
    
    
    prediction = cc_model3.predict([input_list])
    
    if (prediction[0] <= 0.8):
        return 0
    else:
        return 1
    
