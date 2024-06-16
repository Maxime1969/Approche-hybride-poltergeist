import numpy as np
from transformers import RobertaTokenizer, T5ForConditionalGeneration
from Embedding import ClsEmbedding 
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
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

Threshold = [0.04, 0.05, 0.07, 0.08, 0.1, 0.2]
# definie Isolation Forest model
for j in range(epoch):
 for threshold in Threshold:
  listoutliers = []
  for i in range(100):
    modelIF = IsolationForest(n_estimators= 100, max_samples=256, contamination = threshold, max_features = X_data.shape[1])
    modelIF.fit(X_data)
    X_data['anomaly']= modelIF.predict(X_data)
    outliers = [myindex for myindex in X_data.index if X_data.loc[myindex, 'anomaly'] == -1]
    listoutliers.extend(outliers)
    X_data= X_data.drop(columns=X_data.columns[-1])
  alloutliers.append(listoutliers)
 ens_outliers = {item for sublist in alloutliers for item in sublist}
 myoutliers.append((len(ens_outliers), list(ens_outliers)))
number_outliers, list_outliers = zip(*myoutliers)
Number_outliers =list(number_outliers)
List_outliers = list(list_outliers)
max_outliers = max(Number_outliers)
print(max_outliers)
ensemble_outliers = list({item for sublist in List_outliers for item in sublist if len(sublist) == max_outliers})
print(f"ensemble_outliers {ensemble_outliers}")
X_outliers = X_data.loc[ensemble_outliers]

# Trace des clusters
plt.figure(figsize=(8, 6))
plt.plot(range(epoch), number_outliers)
plt.xlabel('Number of epochs')
plt.ylabel('Number of outliers')
plt.title('Visualization of anomalies')
plt.show()




