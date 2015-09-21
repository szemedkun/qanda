# Question Answering System
## Made as a Capstone Project for Galvanize's Data Science Program

### Introduction
The goal of the project is to build a model that understands a set of factual sentences to be able to answer questions based on the facts. For example, 
> Mary is in the bathroom. Mary moved to the Bedroom. John traveled to the hallway.
> Where is Mary?

The model should know that Mary is in the Bedroom.

Data scientists at Facebook reported an impressive work using a special kind of recurrent neural networks (RNN) called long short-term memory (LSTM) [here.](http://arxiv.org/pdf/1502.05698v6.pdf)

Later, Stephen Merity used Keras deep learning library to produce [this.](http://smerity.com/articles/2015/keras_qa.html) His success is due to the clever idea of his which is splitting the factual statements from the question and merging them back together at the end. Can I combine the two clever ideas with the use of pre-trained GloVe word vectors for word ebedding to train a general question answering system? It turns out combining two great ideas results in a better performance.

### Dataset
The dataset I trained my model comes from [facebook's bAbI project.](https://research.facebook.com/researchers/1543934539189348)

I completed this project in just over two weeks time, so not everything is as perfect as I would have liked it to be.