# Question Answering System
## Made as a Capstone Project in Galvanize's Data Science Program

### Introduction
The goal of the project is to build a model that understands a set of factual sentences to be able to answer questions based on the facts. For example, 
> Mary is in the bathroom. Mary moved to the Bedroom. John traveled to the hallway.
> Where is Mary?

The model should know that Mary is in the Bedroom.

This project was motivated by [Stephen Merity's Blog.](http://smerity.com/articles/2015/keras_qa.html)

### Dataset
The dataset I trained my model comes from [facebook's bAbI project.](https://research.facebook.com/researchers/1543934539189348)

### Model

I used a recurrent neural network (RNN) (Long short-term memory (LSTM)) and keras deep learning library to train my model. A high level brief explanation of the model is described below.

I also used pre-trained on wikipedia GloVe word vectors for word embedding in my LSTM model. A high level brief explanation of the GloVe vector representation of words is found below.

#### RNN and LSTM
RNN: 

LSTM:

#### Word Embedding
The two most popular vector representation of words are word2vec and GloVe. Both models use a large corpus to learn co-occurrence statistics of words. They both perform similarly for subsequent machine learning purposes.

1. Word2vec is a predictive model that ... link here?

2. GloVe is a model ... link here?


I completed this project in just over two weeks time, so not everything is as perfect as I would have liked it to be.
