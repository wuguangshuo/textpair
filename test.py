import pandas as pd
from sentence_transformers import evaluation,SentenceTransformer,util

#y由于没划分测试集，就拿这个当测试集了
sentences1,sentences2,scores = [],[],[]
dev_data_raw=pd.read_csv('my_dev.csv').values.tolist()[101:2001]
for data in dev_data_raw:
    sentences1.append(data[1])
    sentences2.append(data[2])
    scores.append(float(data[3]))
evaluator=evaluation.EmbeddingSimilarityEvaluator(sentences1,sentences2,scores)
model = SentenceTransformer('./save')
res=model.evaluate(evaluator)
print(res)
print('finish')

#取出个例子尝试一下
emb1 = model.encode('孩子咳嗽哮喘，坚持凉水洗澡行吗？')
emb2 = model.encode("孩子咳嗽哮喘能凉水洗澡吗？")
cos_sim = util.pytorch_cos_sim(emb1, emb2)
print("Cosine-Similarity:", cos_sim)