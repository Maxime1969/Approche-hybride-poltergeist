from sklearn.utils import resample
from Embedding import ClsEmbedding 
from sklearn.cluster import KMeans
import pandas as pd
import matplotlib.pyplot as plt
from transformers import RobertaTokenizer, T5ForConditionalGeneration
inert = []
#Parameters
batch = 100
#Embedding Model
model_name = 'Salesforce/codet5-small'
tokenizer = RobertaTokenizer.from_pretrained(model_name)
model = T5ForConditionalGeneration.from_pretrained(model_name)
#Data loading
data_ = pd.read_csv("/home/gnagneseraMI/File/apache-ant.csv", sep =',', index_col = "ID")
datafile = "/home/gnagneseraMI/File/apachedata"
embedding = ClsEmbedding(data_, datafile, tokenizer, model, batch)
X_data = embedding.get_matrix()
for i in range(1, 10):
   kmeans = KMeans(n_clusters=i, random_state=42, n_init= 10)
   # Train Model
   kmeans.fit(X_data)
   inert.append(kmeans.inertia_)

#Elbow method  
plt.plot(range(1,10), inert)
plt.title("Elbow Method")
plt.xlabel("Number of Clusters")
plt.ylabel("Inertia")
plt.show()
