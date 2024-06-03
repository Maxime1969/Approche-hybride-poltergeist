from transformers import RobertaTokenizer, T5ForConditionalGeneration
from sklearn.utils import resample
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from torch.optim import Adam
from Embedding import ClsEmbedding 
import MyVae as eav
from MyVae import ClsVAE
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import matplotlib.pyplot as plt
from Codeset import CodeDataset

#Parameters
batch_size = 100
hidden_dim = 500
latent_dim = 250  # Latent space dimension
lr = 1e-3
epochs = 100
beta = 4.0  # Parameter β
Embedding Model
model_name = 'Salesforce/codet5-small'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
#Data loading
data_ = (pd.read_csv('/home/gnagneseraMI/File/apache-ant.csv', sep =',', index_col = "ID"))
data_ = resample(data_, replace = False, n_samples = len(data_), random_state=42)
datafile = "/home/gnagneseraMI/File/apachedata"
embedding = ClsEmbedding(data_, datafile, tokenizer, model, batch_size)
matrix = embedding.get_matrix()
data_train, data_test = train_test_split(matrix, test_size=0.20, random_state=42)
datasetrain = CodeDataset(data_train)
X_dataloader_train = DataLoader(datasetrain, batch_size=batch_size, shuffle=True)
datasetest = CodeDataset(data_test)
X_dataloader_test = DataLoader(datasetest, batch_size=batch_size, shuffle=True)
# Initialiser le modèle et l'optimiseur
vae = ClsVAE(matrix.shape[1], hidden_dim, latent_dim)
optimizer = optim.Adam(vae.parameters(), lr=lr)
 
# train VAE
train_losses = eav.train_vae(vae, X_dataloader_train, optimizer, epochs, X_dataloader_test, beta)

#Loading true positive data
data_positif = (pd.read_csv('/home/gnagneseraMI/File/eif/poltergeists.csv', sep =',', index_col = "ID"))
data_positif = resample(data_positif, replace = False, n_samples = len(data_positif), random_state=42)
datafile_positif = "/home/gnagneseraMI/File/eif/File_Poltergeists"
embeddingpolt = ClsEmbedding(data_positif, datafile_positif, tokenizer, model, batch_size)
matrixpoltergeist = embeddingpolt.get_matrix()
datasetvalue = CodeDataset(matrixpoltergeist)
X_dataloader_evalue = DataLoader(datasetvalue, batch_size=1, shuffle=True)
  
#Loading true negative data
data_fauxpositif = (pd.read_csv('/home/gnagneseraMI/File/eif/FalsePositive.csv', sep =',', index_col = "ID"))
data_fauxpositif = resample(data_fauxpositif, replace = False, n_samples = len(data_fauxpositif), random_state=42)
datafile_fauxpositif = '/home/gnagneseraMI/File/eif/File_FalsePositifs'
embeddingfaux = ClsEmbedding(data_fauxpositif, datafile_fauxpositif, tokenizer, model, batch_size)
matrixfaux = embeddingfaux.get_matrix()
datasetvaluefaux = CodeDataset(matrixfaux)
X_dataloaderfaux_evalue = DataLoader(datasetvaluefaux, batch_size=1, shuffle=True)
  
#detection validation
threshold = [0.05, 0.06, 0.07, 0.08, 0.1]
numbersample = 10
for seuil in threshold:
  anomaly_indices = eav.anomaly_detection(vae, X_anomalous = X_dataloader_evalue, threshold = seuil, numbersample = numbersample)
  print(anomaly_indices)
  anomaly_faux = eav.anomaly_detection(vae, X_anomalous = X_dataloaderfaux_evalue, threshold = seuil, numbersample = numbersample)
  print(anomaly_faux)
  sampletrue = range(1, len( X_dataloader_evalue.dataset) + 1)
  samplefalse = range(1, len( X_dataloaderfaux_evalue.dataset) + 1)
  sample = range(1, numbersample + 2)
  plt.figure(figsize=(8, 6))
  plt.plot(sampletrue, anomaly_indices, label='Probability of reconstruction of true positives',  marker='o', color='black')
  plt.plot(samplefalse, anomaly_faux, label='Probability of reconstruction of false positives', marker='o', color='blue')
  plt.plot(samplefalse, [seuil for _ in samplefalse]  , label='Reconstruction Probability Threshold', color='red')
  plt.xlabel('sample number')
  plt.ylabel('Probability of reconstruction')
  plt.title(' Probability of reconstruction')
  plt.legend()
  plt.grid(True)
  plt.show()
    
