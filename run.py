import pandas as pd
from sentence_transformers import InputExample,SentencesDataset,SentenceTransformer,losses,evaluation
from torch.utils.data import DataLoader
train_data_raw=pd.read_csv('my_train.csv').values.tolist()[:10]
dev_data_raw=pd.read_csv('my_dev.csv').values.tolist()[:10]

#处理train数据
train_data=[]
for data in train_data_raw:
    train_data.append(InputExample(texts=[data[1],data[2]],label=float(data[3])))
print('训练数据处理完成')

#处理验证集
sentences1,sentences2,scores = [],[],[]
for data in dev_data_raw:
    sentences1.append(data[1])
    sentences2.append(data[2])
    scores.append(float(data[3]))
evaluator=evaluation.EmbeddingSimilarityEvaluator(sentences1,sentences2,scores)
#构建模型
model=SentenceTransformer('bert-base-chinese')
train_dataset=SentencesDataset(train_data,model)
train_dataloader=DataLoader(train_dataset,shuffle=True,batch_size=4)
train_loss=losses.CosineSimilarityLoss(model)
model.fit(train_objectives=[(train_dataloader,train_loss)],epochs=10,warmup_steps=100,evaluator=evaluator,evaluation_steps=100,output_path='./save')














