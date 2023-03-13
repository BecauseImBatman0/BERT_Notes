# BERT_Notes
BERT学习笔记+文本分类实现

## BERT
BERT是一个基于Transformer的预训练模型，利用了Transformer的编码器进行叠加，从而在许多NLP任务上取得了sota的效果。  
![图片](https://user-images.githubusercontent.com/126166790/224623896-681be04c-ea03-46c9-b6df-89bf268b6809.png)  


它的核心思想是利用了双向思维，论文中说到语言模型的主要限制是单向性，而这限制了预训练时可选择的架构。文中提到了BERT比简单的单一方向以及从左到右结合从右到左（concatenation）效果都要好，并且利用了两种预训练的任务（完形填空和下句判断），这样得到的预训练模型只需要微调（fine-tuning）就能在下游任务中得到不错的效果。

## 预训练任务1——完形填空（Masked LM）
作者将文本随机的15%的内容用[MASK]进行处理，有80%的几率被真实地替换为[MASK],有10%的几率随机替换为其他词，还有10%不改变，这是因为考虑到实际的下游任务中并没有出现[MASK].

## 预训练任务2——下句判断（Next Sentence Prediction NSP）
可以输入1个句子也可以输入2个句子，用来预测第二个句子是不是位于第一个句子之后，用3个embedding对输入文本进行表示。

文章的后半部分主要是关于在各个任务上的实验表现以及消融实验，在附录中详细介绍了预训练和微调流程。  

## 利用BERT做一个文本分类任务
使用pytorch和hugging face的transformers库，数据集为sentiment-analysis-on-movie-reviews。  
>🤗抱抱脸：<https://github.com/huggingface/transformers>  

数据集介绍:大致长这样，在每个句子后面的数字是情感，情感有5种：0.negative  1.somewhat negative  2.neutral  3.somewhat positive  4.positive  
![Uploading 图片.png…]()

### 主要步骤为：
### 1、文本嵌入表示  
### 2、将5种情感转为独热码表示  
### 3、搭建基于BERT的模型  
### 4、训练
在初始数据处理阶段还对数据集进行了划分，先总体打乱，然后将80%作为训练集，剩下20%作为测试集。关于文本的嵌入，使用的是transformers库提供的AutoTokenizer，非常方便。

使用独热编码即可。

模型构建：比较简单，加载BERT预训练模型，然后加了一个dropout，以及一个Linear层。
损失函数使用交叉熵，但是适合效果没太好，优化器用Adam。完整代码如下。
