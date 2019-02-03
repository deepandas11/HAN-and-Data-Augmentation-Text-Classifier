# ECE539 Academic Project: Exploring Hierarchical Attention based Deep Neural Networks for Toxic Comments Classification

## Introduction

Text classification is a task in NLP in which predefined categories are assigned to text documents. The two widely used approaches can be represented as follows: 

![text-classification-workflow1](https://user-images.githubusercontent.com/19747416/52180628-5b3c9300-27ae-11e9-9180-dd8b87352b4a.png)

One major difference in classical machine learning and deep learning approaches is that in the deep learning approach, feature extraction and classification is carried out together. 

Proper feature representation is an important task in any Machine Learning task. In this case, we have text comments that are made of sentences and in return, sentences are made of words. A big challenge for NLP is to represent text in a proper format. We will be using **Vector space models** to represent the text corpus. The other inefficient method is the **Standard Boolean method.**

Before we explore the Deep Learning models, we ensure that the dataset is understood well and that it is suitable for applying Deep Learning based methods


[1. Data Visualization and Feature Analysis](https://github.com/deepandas11/HAN-and-Data-Augmentation-Text-Classifier/blob/master/notebook1-data-visualization%20and%20feature%20analysis.ipynb)
Tasks accomplished in this notebook:
  - Visualizing categorization of data
  - Visualizing categorization within toxic corpus
  - Analysing the overlap across the various categories of toxicity
  - Feature analysis 
  - Understanding the importance of certain features in different classes of toxicity
  - Adversarial Validation to ensure we can use Cross validation on our classification model.

[2. Baseline LSTM Model with Data Augmentation](https://github.com/deepandas11/HAN-and-Data-Augmentation-Text-Classifier/blob/master/notebook3-baseline-lstm-data-augmentation.ipynb)

[3. Hierarchical Attention Network Model](https://github.com/deepandas11/HAN-and-Data-Augmentation-Text-Classifier/blob/master/notebook4-han-with-augmented-data.ipynb)


