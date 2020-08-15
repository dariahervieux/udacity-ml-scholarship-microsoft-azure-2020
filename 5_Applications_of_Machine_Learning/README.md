- [Deep learning](#deep-learning)
  - [Applications](#applications)
- [Specialized Cases of Model Training](#specialized-cases-of-model-training)
  - [Similarity learning](#similarity-learning)
  - [Text classification](#text-classification)
  - [Feature learning](#feature-learning)
    - [Application of feature learning](#application-of-feature-learning)
      - [Image classification with CNN](#image-classification-with-cnn)
      - [Image search with Autoencoder](#image-search-with-autoencoder)
  - [Anomaly detection](#anomaly-detection)
  - [Forecasting](#forecasting)
- [Resources](#resources)

# Deep learning

Characteristics:
* It can be very computationally expensive, especially as model complexity increases
* It works well with massive amounts of training data
* Effective on different data types (numbers, images, text)
* It is capable of extracting features automatically
* Non-parametric approach helps to learn arbitrarily complex functions
* They can learn complex patterns without explicitly seeing those patterns.
* Can be distributes for parallel training
* Can learn time-related patterns (RNN)
* Capable of reaching on-par performance with certain human activities
  
## Applications

* Language translation
* Image recognition
* Speech recognition
* Forecasting
* Predictive analytics
* Autonomous vehicles

Examples of applications in text analythics:
* semantics - getting the meaning of the text
* sentiment
* summarization - extraction of key phrases, entities, topics
* classification
* clustering - identifying similarity between documents/terms
* search

# Specialized Cases of Model Training

Special categories of problems:
* Similarity training - learns from examples using similarity function, *supervised learning*
* Feature learning/representation learning - transforming input into another input more appropriate for problem solving, *supervised learning* (using labeled data) and *unsupervised learning* (using unlabeled data)
* Anomaly detection - classification of unbalanced data (there is much more items of one class than another), *supervised learning* (leanrs from data labelled normal/abnormal) and *unsupervised learning* (learns from unlabelled data assuming most entities are normal)

## Similarity learning

Similarity learning is closely related to classification and regression (supervised learning), but uses a  different type of objective function.
* Approached as a classification problem: the similarity function maps pairs of entities (users + items) to a finite number of similarity levels (ranging from 0/1 to any number of levels). The output is discrete (levels of similarity).
* Approached as regression problem: the similarity function maps pairs of entities (users + items) to numerical value (similarity score). The output is continues. 
* Variation of regression approach: *ranking similarity learning*, where exact measure is replaced with ordering measure. This a better fit for large-scale real-life problems, since if helps to increase the performance

Often used in recommendation systems. Also  often used in verification problems (speech, face, ..)

The main aim of a recommendation system is to recommend one or more items to users of the system. Examples of an item to be recommended, might be a movie, restaurant, book, or song. In general, the user is an entity with item preferences such as a person, a group of persons, or any other type of entity.

Principal approaches to recommender systems:
1. The *content-based* approach, which makes use of features for both *users and items*. Users can be described by properties such as age or gender. Items can be described by properties such as the author or the manufacturer. Typical examples of content-based recommendation systems can be found on social matchmaking sites.
2. The *Collaborative filtering* approach, which uses only *identifiers* of the users and the items. It is based on a *matrix of ratings* given by the users to the items. The main source of information about a user is the list the items they’ve rated and the similarity with other users who have rated the same items. The rating can also be implicit: purchase history, browsing history.

Types of predictions:
1. *Prediction of ratings* - the model calculates how a user will react to a particular item, given the training data. The input data for scoring must provide *both a user and the item to rate*.
2. *Recommendations for users* - 

The [*SVD recommender*](https://docs.microsoft.com/en-us/azure/machine-learning/algorithm-module-reference/train-svd-recommender) module in Azure Machine Learning designer is based on the *Singular Value Decomposition* algorithm. It uses identifiers of the users and the items, and a matrix of ratings given by the users to the items. It’s a typical example of collaborative recommender.
Note: In other pipelines, for selecting a list of columns from a dataset, we could have used the Select Columns from Dataset prebuilt module. This one returns the columns in the same order as in the input dataset. This time we need the output dataset to be in the format: user id, movie name, rating.This column order is required at the input of the Train SVD Recommender module.
SVD recommender hyper parameters:
* Number of factors (default 200). This option specify the number of factors to use with the recommender. With the number of users and items increasing, it’s better to set a larger number of factors. But if the number is too large, performance might drop
* Number of recommendation algorithm iterations(default  30). This number indicates how many times the algorithm should process the input data. The higher this number is, the more accurate the predictions are. However, a higher number means slower training. The default value is 30.
* Learning rate (default 0.005) - the size of the step in the learning process

## Text classification

In text classification scenarios, the goal is to assign a piece of text, such as a document, a news article, a search query, an email, a tweet, support tickets, customer feedback, user product review, to predefined classes or categories. Some examples of text classification applications are: categorizing newspaper articles into topics, organizing web pages into hierarchical categories, spam email filtering, sentiment analysis, predicting user intent from search queries, support tickets routing, and customer feedback analysis.

Before we can do text classification, the text first needs to be translated into some kind of numerical representation—a process known as text embedding. The resulting numerical representation, which is usually in the form of vectors, can then be used as an input to a wide range of classification algorithms.

![Training classification model with text](https://cloudworkshop.blob.core.windows.net/cognitive-deep-learning/Whiteboard%20design%20session/images/Whiteboarddesignsessiontrainerguide-CognitiveServicesanddeeplearningimages/media/image2.png)

![Predicting a classification from text](https://cloudworkshop.blob.core.windows.net/cognitive-deep-learning/Whiteboard%20design%20session/images/Whiteboarddesignsessiontrainerguide-CognitiveServicesanddeeplearningimages/media/image3.png)

Vectorization techniques:
* [TF-IDF](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) - term frequency–inverse document frequency

Extract N-gram feature from text
Feature hashing

## Feature learning

Examples of feature learning;
* supervised approaches - new features are learned from labelled data:
  * Dataset contains multiple categorical features with high cardinality => use supervised encoding with Deep Learning (where feature embedding happens withing the hidden layers)
* unsupervised approachas -
  * clustering 
  * Deep Learning autoencoding
  * PCA
  * independent component analysis
  * matrix factorization

### Application of feature learning

Applications:
* image classification
* image search
* feature embedding (categorical features with high cardinality)

#### Image classification with CNN

 Convolutional Neural Networks (CNNs) consists:
 * convolution layer (1 to N) - to learn local patterns
 * max-pooling layer (1 to N, goes in pair with conv layer) - to down-sample the data and make it *translation invariant*.
 * dense layer - densely connected layer to produce the final result, to densely connect all outputs and learn the classification.
  
  CNN applies a matrix on the input data called *kernel*. There is a list of well-known kernels, one for each specific image tasks: edge detection, sharpening, etc..

  One can use CNN layers to learn some *local* patterns. Local patterns are *translation invariant*: once the network is capable to recognize local pattern in one place, it can recognize in the other. It is different from Densly Connected NN which learn global patterns. CNN has a capability to learn the hierarchies of patterns. 

#### Image search with Autoencoder

Autoencoder - a neural network for *unsupervised learning*, it trains to reproduce as accurate as possible *its own inputs*, the very values data are like labels.

Typical structure:
* encoder:
  * large input layer
  * each following layer is narrower than the previous one until the middle layer
* middle layer - the layer with the smallest width, a.k.a. feature vector - compressed representation of the input (compression in the sense of dimensionality reduction, not data compression)
* decoder:
  * each following layer encreases in width till the original size of the input

As a result of its inner structure, ths the left part (ecoder) can be used to embed inputs into *feature vectors*.
Using a special function (measuring distance between vectors), we can find similar images (the threshorld of the distance value is below a certain value).

## Anomaly detection

Two classes are severe unbalanced, abnormal class is much much lower than the normal class.

Approaches:
* supervised - binary classification with input data labeled as normal/abnormal
* unsupervised - clustering: detection of 2 groups: normal cases and abnormal

Applicaion:
* machinery maintenance, failure prevention (industrial machinery; failure is detected when the model gives the result above the learned "failure" threshold)
* fraud detection
* intrusion detection
* anti-malweare protection
* data preparation - identification of outliers

## Forecasting

A typical example of a forecasting problem: given a set of ordered (in time, or just orderable items on a column that gives an explicit order) data points (such as sales data over a series of dates), we want to predict the next data points in the series for a defined time window (such as what sales will look like next week).

Types of algorithms used to perform predictions:
* AutoRegressive Integrated Moving Average (ARIMA)
* Multi-variate regression - apart time (order) column, we can take other features into consideration
* Prophet - works best for time-series that have seasonal effect
* ForecastTCN (Temporal Convolutional Networks) - one dimentional convolutional network, this algorithm has a longer "memory" than other approaches
* Recurrent Neural Networks - a classical netork with additional connections between nodes

# Resources

* [Deep Learning, by Ian Goodfellow, Yoshua Bengio, Aaron Courville](https://www.deeplearningbook.org/contents/mlp.html)- [Resources](#resources)