"""
Test POST to Live API

Test POST to Live API and print results

By: Julian Bolivar
Version: 1.0.0
Date:  2023-05-30
Revision 1.0.0 (2023-05-30): Initial Release
"""
import requests
import json

#Live API URL
url = "https://census-data-processor.onrender.com/inference"


# explicit the sample to perform inference on
sample =  { 'age':35,
            'workclass':"Private", 
            'fnlgt':654321,
            'education':"Doctorate",
            'education_num':16,
            'marital_status':"Separated",
            'occupation':"Exec-managerial",
            'relationship':"Not-in-family",
            'race':"Black",
            'sex':"Female",
            'capital_gain':0,
            'capital_loss':0,
            'hours_per_week':60,
            'native_country':"United-States"
            }

data = json.dumps(sample)

# post to API and collect response
response = requests.post(url, data=data )

# display output: response will show sample details + model prediction added
print("response status code", response.status_code)
print("response content:")
print(response.json())
