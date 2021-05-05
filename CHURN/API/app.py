from flask import Flask, request
import pandas as pd
import pickle
import os
from churn_prep import ChurnPreprocessing

modelo = pickle.load(open("CHURN/model.pkl","rb")) 

app = Flask(__name__)

@app.route('/predict',methods=['POST'])
  
# end-point da predição 
def predict():
    data_json = request.get_json(force=True)

    # coleta o dado em json e transformar em dataframe
    df_raw = pd.DataFrame( data_json )

    # instancia classe para tratar o dado
    pipeline = ChurnPreprocessing()

    # aplica as transformações no dado
    df1 = pipeline.data_preparation(df_raw)

    # predição do modelo 
    pred = modelo.predict(df1)

    # resposta da predição 
    df_raw['prediction'] = pred

    return df_raw.to_json(orient='records')
    

if __name__ == '__main__':
    # inicia o debug
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port)