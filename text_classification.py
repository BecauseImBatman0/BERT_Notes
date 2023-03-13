from transformers import AutoTokenizer, BertModel
import torch
import pandas as pd
from torch.utils.data import DataLoader,Dataset
import torch.nn as nn
import random

def load_data(path):
    data = pd.read_csv(path, delimiter='\t')
    sentence=data["Phrase"]
    sentiment=data["Sentiment"]     #对于sentiment 0.negative  1.somewhat negative  2.neutral  3.somewhat positive  4.positive
    return sentence.tolist(),sentiment.tolist()

def text2embedding(text):
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    embeding = tokenizer(text, padding="max_length", truncation=True, max_length=MAX_LENGTH)  # 超出最大长度时截断
    return embeding

class BERT1(nn.Module):
    def __init__(self):
        super(BERT1,self).__init__()
        self.pretrain_model=BertModel.from_pretrained("bert-base-uncased",return_dict=False)
        self.dropout=nn.Dropout(0.3)
        self.linear=nn.Linear(768,5) #BERR预训练模型输出768维张量
        self.softmax=nn.Softmax(dim=-1)
    def forward(self,input_ids,attention_mask,token_type_ids):
        _,output1=self.pretrain_model(input_ids,attention_mask,token_type_ids)
        output2=self.dropout(output1)
        output3=self.linear(output2)
        output=self.softmax(output3)
        return output

class DataProcessor(Dataset):
    def __init__(self, sentence,sentiment):
        self.sentence=sentence
        self.sentiment=sentiment

    def __getitem__(self, index):

        return {
            'ids': torch.tensor(self.sentence["input_ids"][index], dtype=torch.long),
            'mask': torch.tensor(self.sentence["attention_mask"][index], dtype=torch.long),
            'token_type_ids': torch.tensor(self.sentence["token_type_ids"][index], dtype=torch.long),
            'targets': torch.tensor(torch.tensor(self.sentiment[index]), dtype=torch.float)
        }

    def __len__(self):
        return len(self.sentiment)#之前大意了，写成len(self.sentence)，然而self.sentence是一个字典,导致len(self.sentence)只有3.....


def train(epoch):
    model.train()

    for _, data in enumerate(dataloader,0):

        ids = data['ids'].to(DEVICE, dtype=torch.long)
        mask = data['mask'].to(DEVICE, dtype=torch.long)
        token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
        targets = data['targets'].to(DEVICE, dtype=torch.float)

        outputs = model(ids, mask, token_type_ids)

        optimizer.zero_grad()
        loss = loss_fn(outputs, targets)
        if _ % 50 == 0:
            print(f'Epoch: {epoch}, Loss:  {loss.item()}')

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

def test():
    test_sentence_embedding = text2embedding(test_sentence)
    test_sentiment_onehot = [nn.functional.one_hot(torch.tensor(i), num_classes=5) for i in test_sentiment]
    dataset_test = DataProcessor(test_sentence_embedding, test_sentiment_onehot)
    dataloader_test = DataLoader(dataset_test, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)  # shuffle=True 随机洗牌
    count=0
    model.eval()
    with torch.no_grad():
        for data in dataloader_test:
            ids = data['ids'].to(DEVICE, dtype=torch.long)
            mask = data['mask'].to(DEVICE, dtype=torch.long)
            token_type_ids = data['token_type_ids'].to(DEVICE, dtype=torch.long)
            targets = data['targets'].to(DEVICE, dtype=torch.float)

            outputs = model(ids, mask, token_type_ids)

            for i in range(len(outputs)):
                idx=torch.argmax(outputs[i])
                if targets[i][idx]==1:
                    count+=1
    return count/len(test_sentiment)

MAX_LENGTH=64
EPOCHS=3
LEARNING_RATE = 0.05
BATCH_SIZE=32   #调成64显存就炸 以后一定要搞一张好的显卡
DEVICE="cuda" if torch.cuda.is_available() else "cpu"

path1="D:\\新-学习文件\\NLP\\sentiment-analysis-on-movie-reviews\\train.tsv"

train_sentence,train_sentiment=load_data(path1)

'''划分训练集和测试集'''
merged_list = [train_sentence[i] + str(train_sentiment[i]) for i in range(len(train_sentence))]

random.shuffle(merged_list)#打乱

train_set = merged_list[:int(0.8 * len(merged_list))]#训练集80%
test_set = merged_list[int(0.8 * len(merged_list)):]#测试集20%

train_sentence = []
train_sentiment = []

for string in train_set:
    train_sentence.append(string[:-1])
    train_sentiment.append(int(string[-1]))

test_sentence = []
test_sentiment = []

for string in test_set:
    test_sentence.append(string[:-1])
    test_sentiment.append(int(string[-1]))



train_sentence_embedding=text2embedding(train_sentence)



train_sentiment_onehot=[nn.functional.one_hot(torch.tensor(i),num_classes=5) for i in train_sentiment]

dataset=DataProcessor(train_sentence_embedding,train_sentiment_onehot)

dataloader=DataLoader(dataset,batch_size=BATCH_SIZE,shuffle=True,num_workers=0)   #shuffle=True 随机洗牌

model=BERT1()
model.to(DEVICE)
loss_fn=nn.CrossEntropyLoss()

optimizer=torch.optim.Adam(params = model.parameters(),lr=LEARNING_RATE)

for epoch in range(EPOCHS):
    train(epoch)
torch.save(model,"bert_text_classification1.pth")
print("accuracy",test())



