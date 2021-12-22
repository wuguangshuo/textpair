import numpy as np
from sklearn.metrics import classification_report
from sentence_transformers import SentenceTransformer, util
import pandas as pd
model_path = './save_model'
model = SentenceTransformer(model_path)

dev_data_raw=pd.read_csv('my_dev.csv').values.tolist()
sentences1,sentences2,scores = [],[],[]
for data in dev_data_raw:
    sentences1.append(data[1])
    sentences2.append(data[2])
    scores.append(float(data[3]))
s1 = np.array(sentences1)
s2 = np.array(sentences2)
embedding1 = model.encode(s1, convert_to_tensor=True)
embedding2 = model.encode(s2, convert_to_tensor=True)
pre_labels = [0] * len(s1)
predict_file = open('predict.txt', 'w',encoding='utf-8')
for i in range(len(s1)):
    similarity = util.cos_sim(embedding1[i], embedding2[i])
    if similarity > 0.5:
        pre_labels[i] = 1
    predict_file.write(s1[i] + ' ' +
                       s2[i] + ' ' +
                       str(scores[i]) + ' ' +
                       str(pre_labels[i]) + '\n')
print(classification_report(scores, pre_labels))
predict_file.close()
