This repository contains ipynb with experiments on binary text classification(russian language). Three approaches were used : pretrained glove, rnn, bert. Some comparison and analysis provided at the bottom of ipynb.

In addition to that repository contains rest api deployment code for the best model (in my case that was bert).


# Running server:
    
  1. install required python packages ```pip install -r requirements.txt```
  2. load weights for neural network https://drive.google.com/file/d/1a_8w7N_Kd5fil1CXuY-32c4hnE51oP8x/view?usp=sharing and put it in folder as 'bert_pretrained.pt'
  3. run REST API server ```python flask_test_bert.py```

# REST API(Postman) :
  - Install postman 
  - Use POST method with address: ```http://127.0.0.1:5500/predict```
  - In body create JSON with like ```{"text":"python это топчик"}```
  - Receive json with predicted class and probability

# REST API(curl)
  - Instal curl
  - Write `curl --location --request POST 'http://127.0.0.1:5500/predict'  --header 'Content-Type: application/json' 
 --data-raw '{
    "text": "python это топчик"
}'`
        

