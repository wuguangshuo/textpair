from tqdm import tqdm
import numpy as np
import faiss
import pandas as pd
from sentence_transformers import SentenceTransformer

dev_data_raw=pd.read_csv('my_dev.csv').values.tolist()[:200]
ALL = []
for data in tqdm(dev_data_raw):
    ALL.append(data[1])
    ALL.append(data[2])
ALL = list(set(ALL))

model = SentenceTransformer('./save')
DT = model.encode(ALL)
DT = np.array(DT, dtype=np.float32)

# https://waltyou.github.io/Faiss-Introduce/
print(DT[0].shape[0])
index = faiss.IndexFlatL2(DT[0].shape[0])   # build the index
print(index.is_trained)
index.add(DT)                  # add vectors to the index
print(index.ntotal)

k = 10                          # we want to see 10 nearest neighbors
aim = 0
D, I = index.search(DT[aim:aim+1], k) # sanity check
print(I)
print(D)
print([ALL[i]for i in I[0]])