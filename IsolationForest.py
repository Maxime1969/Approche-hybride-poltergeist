import numpy as np
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from Embedding import ClsEmbedding 
import pandas as pd
from sklearn.ensemble import IsolationForest
#Parametre
batch = 100
#Creation du modele embedding
model_name = 'Salesforce/codet5-small'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
#Chargement des donnees
data_ = pd.read_csv("/home/gnagneseraMI/File/apache-ant.csv", sep =',', index_col = "ID")
datafile = "/home/gnagneseraMI/File/apachedata"
embedding = ClsEmbedding(data_, datafile, tokenizer, model, batch)
X_data = embedding.get_matrix()

alloutliers = []

# definir le modele IsolationForest
Seuil = [0.04, 0.05, 0.07, 0.08, 0.1, 0.2]

for seuil in Seuil:
  listoutliers = []  
  for i in range(100):
    model = IsolationForest(n_estimators= 100, max_samples=256, contamination = seuil, max_features = X_data.shape[1])
    model.fit(X_data)
    X_data['anomaly']= model.predict(X_data)
    outliers = [myindex for myindex in X_data.index if X_data.loc[myindex, 'anomaly'] == -1]
    listoutliers.extend(outliers)
    X_data= X_data.drop(columns=X_data.columns[-1])
  alloutliers.append(listoutliers)
  
ens_outliers = {item for sublist in alloutliers for item in sublist} 

list_outliers = list(ens_outliers)
X_outliers = X_data.loc[list_outliers]

