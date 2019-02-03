# ECE539 Academic Project: Exploring Hierarchical Attention based Deep Neural Networks for Toxic Comments Classification

## Introduction

Text classification is a task in NLP in which predefined categories are assigned to text documents. The two widely used approaches can be represented as follows: 

![text-classification-workflow1](https://user-images.githubusercontent.com/19747416/52180628-5b3c9300-27ae-11e9-9180-dd8b87352b4a.png)

One major difference in classical machine learning and deep learning approaches is that in the deep learning approach, feature extraction and classification is carried out together. 

Proper feature representation is an important task in any Machine Learning task. In this case, we have text comments that are made of sentences and in return, sentences are made of words. A big challenge for NLP is to represent text in a proper format. We will be using **Vector space models** to represent the text corpus. The other inefficient method is the **Standard Boolean method.**

To represent the various features in the text corpus we use the [GloVE](https://nlp.stanford.edu/projects/glove/) Embedding vectors. GloVe is an unsupervised learning algorithm for obtaining vector representations for words. Training is performed on aggregated global word-word co-occurrence statistics from a corpus, and the resulting representations showcase interesting linear substructures of the word vector space.

We choose GloVE and not random embedding as in this embedding model, there are a few interesting properties:

 - Euclidean distance between expected similar words is minimized. This resembles the Nearest Neighbor method.

 - This nearest neighbor method to produce a single scalar quanitifying the relatedness of two words can be problematic as there may be other intricate relationships that exist across words.  For example, man may be regarded as similar to woman in that both words describe human beings; on the other hand, the two words are often considered opposites since they highlight a primary axis along which humans differ from one another.  A natural and simple candidate for an enlarged set of discriminative numbers is the vector difference between the two word vectors. GloVe is designed in order that such vector differences capture as much as possible the meaning specified by the juxtaposition of two words.


## [1. Data Visualization and Feature Analysis](https://github.com/deepandas11/HAN-and-Data-Augmentation-Text-Classifier/blob/master/notebook1-data-visualization%20and%20feature%20analysis.ipynb)

Before we explore the Deep Learning models, we ensure that the dataset is understood well and that it is suitable for applying Deep Learning based methods. Tasks accomplished in this notebook:
  - Visualizing categorization of data
  - Visualizing categorization within toxic corpus
  - Analysing the overlap across the various categories of toxicity
  - Feature analysis 
  - Understanding the importance of certain features in different classes of toxicity
  - Adversarial Validation to ensure we can use Cross validation on our classification model.

## [2. Baseline LSTM Model with Data Augmentation](https://github.com/deepandas11/HAN-and-Data-Augmentation-Text-Classifier/blob/master/notebook3-baseline-lstm-data-augmentation.ipynb)

Recurrent Neural Networks and its derivatives is a well-researched topic in NLP. RNNs use internal memory to process input sequences. Humans understand each word based on our understanding of a set of previous words because our thoughts are persistence. Traditional Neural networks cant handle this issue, but RNNs allow loops in them, allowing information to exist. The following chain like structure enables RNNs to initimately understand sequences and lists.

**Long Term Dependencies**


![rnn-unrolled](https://user-images.githubusercontent.com/19747416/52180944-af954200-27b1-11e9-9260-10d53f60e2e3.png) 

[Reading: Colah's Blog](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)


[3. Hierarchical Attention Network Model](https://github.com/deepandas11/HAN-and-Data-Augmentation-Text-Classifier/blob/master/notebook4-han-with-augmented-data.ipynb)


