import torch as pt
import sys
import os
import numpy as np
import pandas as pd
import chardet
import gc
#Paramétrage
if pt.cuda.is_available():
  device = "cuda"
  pt.cuda.empty_cache()
  # Activez le GPU.
  pt.cuda.set_device(0)
else:
  device ="cpu"
gc.collect()
class ClsEmbedding():
  def __init__(self, data, datafile, tokenizer, model, batch):
     #Données
     self.data = data
     self.datafile = datafile
     #Création du modèle CodeT5 pour l'embedding
     self.tokenizer = tokenizer
     self.model = model.to(device)
     self.stepbatch = 0
     self.batch = batch
     self.matrix_data= pd.DataFrame(columns=['Matrix'])
  
  #Embedding
  def get_embedding(self):
    
    X_data = self.data.iloc[:, 0].tolist()
    #ID_data = data.index.tolist()
    if len(X_data)%self.batch == 0:
      self.stepbatch = int((len(X_data)/self.batch))
    else:
      self.stepbatch = int((len(X_data)//self.batch)) + 1
    
    for i in range(int(self.stepbatch)):
      batch_samples =[]
      generator_ = []
      batch_ID = []
      if len(X_data) - i*(self.batch + 1) <= self.batch:
        batch_samples = X_data[i*(self.batch + 1): len(X_data)]
        #batch_ID = ID_data[i*(self.batch + 1): len(X_data)]
      else:
        batch_samples = X_data[i*(self.batch + 1):i*(self.batch + 1) + self.batch]
        #batch_ID = ID_data[i*(self.batch + 1):i*(self.batch + 1) + self.batch]
      
      for batchs in batch_samples:  # Loop with step of 'batch'
        if os.path.exists(self.datafile + "/" + batchs):
          
          with open(self.datafile + "/" + batchs, 'rb') as file:
            detected_encoding = chardet.detect(file.read())['encoding']
          with open(self.datafile + "/" + batchs, "r", encoding= detected_encoding) as fichier:
            contenu = fichier.readlines()
            if contenu!=[]:
              batch_ID.append("2024000" + str(i+ 1))
              inputs = self.tokenizer.encode_plus(contenu, padding='longest', truncation=True, return_tensors='pt')
              inputs = inputs.to(device)
              outputs = self.model(inputs.input_ids, attention_mask=inputs.attention_mask, decoder_input_ids = inputs['input_ids'], output_hidden_states=True )
              embedding = outputs.encoder_last_hidden_state
              embedding = embedding.to(device)
              mean = pt.mean(embedding, dim=(1,2))
              std = pt.std(embedding, dim=(1,2))
              normalized_embedding = (embedding - mean) / std
              normalized_embedding = normalized_embedding.to(device)
              reduced_normalized_embedding = pt.mean(normalized_embedding, dim=0).to(device)
              fused_normalized_embeddings = pt.mean(reduced_normalized_embedding, dim=0).to(device)
              x_normalized = (fused_normalized_embeddings - fused_normalized_embeddings.min()) / (fused_normalized_embeddings.max() - fused_normalized_embeddings.min())
              generator_.append(x_normalized)
        else:
          print("Fichier introuvable")
      #print("ID", len(batch_ID))
      #print("File", len(generator_))
      if len(batch_ID) == len(generator_):
        df = pd.DataFrame({'Identity': batch_ID, 'Matrix': generator_})  
        yield df
      
  def get_matrix(self):
    matrix_data = []
    generator = self.get_embedding()
    for j, data_X in enumerate(generator):
      if j == 0:
        matrix_data = data_X.iloc[:, 1:]
      
      else:
        if j > 0:
          matrix_data = pd.concat([matrix_data, data_X.iloc[:, 1:]], axis = 0)
      
    #print(matrix_data)
    flattened_data = matrix_data['Matrix'].apply(lambda x: x.reshape(-1))
    # Créez un nouvel ensemble de données avec les vecteurs aplaties
    new_matrix_data = pd.DataFrame(flattened_data.tolist())
    return new_matrix_data


  
