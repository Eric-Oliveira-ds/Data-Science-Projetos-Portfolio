import pickle

class ChurnPreprocessing ( object ):
    def __init__(self):
        self.encoder = pickle.load(open("CHURN/encoder.pkl","rb"))
        self.min_max = pickle.load(open("CHURN/min_max.pkl","rb"))
        
        def data_preparation (self, df):
            
            df = self.encoder.transform(df.values)
            df = self.min_max.transform(df.values)
            
            return df 
            
            

        
        

