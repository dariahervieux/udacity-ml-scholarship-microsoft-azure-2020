- [Supervised Learning](#supervised-learning)
  - [Classification](#classification)
  - [Regression](#regression)
    - [Metrics](#metrics)
      - [Linear regression](#linear-regression)
  - [Classification/regression algorithms' hyperparameters](#classificationregression-algorithms-hyperparameters)
    - [Classic algorithms](#classic-algorithms)
    - [Neural networks](#neural-networks)
    - [Decision forest](#decision-forest)
  - [Automated ML (Azure)](#automated-ml-azure)
- [Unsupervised Learning](#unsupervised-learning)
- [Semi-supervised learning](#semi-supervised-learning)
  - [Clustering](#clustering)

# Supervised Learning

## Classification

The result of the classification is *categorical or discrete*.

Input types:
* tabular
* image
* sound
* text

Classification types:
* binary - choice between 2 categories
* multi-class single label - choice between multiple classes, item belongs to only one class
* multi-class multi-label - choice between multiple classes, item can belong from one to several categories.

Examples of classification problems:
* computer vision
* speech recognition
* bio-identification
* document classification
* sentiment analysis
* credit scoring in banking
* anomaly detection

Binary classification algorithms:
* SVM (Support Vector Machines) - under 100 features, linear model
* Two-class averaged perception - fast training, linear model
* Two-class decision forest - ensemble algorithm, accurate, linear model
* Two-class  logistic regression - fast training times; linear model
* Two-class  Boosted Decision Tree  - accurate,  fast training, linear model
* Two-class Neural Network - accurate, long training times
  
Multi-class single label classification algorithms:
* Multi-class  logistic regression - fast training times; linear model
* Multi-class Neural Network - accurate, long training times
* Multi-class decision forest - ensemble algorithm (build multiple decision trees and votes on the most popular output class), accurate, linear model
* Multi-class  Boosted Decision Tree  - non-parametric, fast training, scalable
* One-vs-All Multiclass - trains a several binary classifiers where each classifier predicts one of the end classes, the class with the higher probability or a score across binary classifier is chosen as a predicted class

  
## Regression

In a regression problem, the output is numerical or continuous.

Input types:
* tabular
* image
* sound
* text

Examples of regression problems:
* housing prices
* customer churn
* customer lifetime value
* forecasting (time-series)
* anomaly detection

Common machine learning algorithms in Azure for regression problems:
* Linear Regression - fast training, linear model
* Decision Forest Regression - 
  * ensemble algorithm using multiple decision trees
  * each tree output a distribution as a prediction
  * all distibutions are aggregated to find the closest combines distribution
  * Accurate, fast training times
* Neural Net Regression
  * label column must be numerical
  * fully contected network : input layer +  hidden layer + output layer
  * Accurate, long training times

### Metrics

#### Linear regression

Approaches:
* Ordinary least squares - sum of the squared of distance from the actual value to the predicted line, it fits the model by minimazin the squared error.
* Gradient descent - minimize the amount of error on each step of the model training process
  
## Classification/regression algorithms' hyperparameters

### Classic algorithms

Hyperparameters:
* Optimization tolerance - controls when to stop iterations: if the improvement is less than the specified threshold, the training stops
* L2 Regularization weight - a method to prevent overfitting by penalizing models with extrem coefficient weights: how much to penalize a model on each interaction

### Neural networks

Hyperparameters:
* number of hidden nodes - customizing the number of hidden nodes
* learning rate - the number of interations before correction is made
* number of learning iterations - the number of times the algorithm should process the training data

Example on NN for multi-class classification:
* input layer
* hidden layer
* output layer

### Decision forest

Hyperparameters:
* Resampling method - method to create individual trees
* Number of decision trees - maximum number of trees that can be created
* Depth of the tree - Max depth of any created tree
* Number of random splits per node - the number of splits to use when building each node of the tree
* Minimum number of samples per leaf node - the number of samples to create a terminal node

## Automated ML (Azure)

Challenges of conventional ML process:
* selecting features
* picking algorithms suitable for the task
* hyperparameters tuning
* selecting the right evaluation metrics

Automated ML - automated exploration of combinations needed to successfully produce a trained model.
The resulting model can be either deployed or refined further.

AutoML inputs:
* dataset
* target metric
* constraints (time/cost)

Official AutoML [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/concept-automated-ml).
Automatic featurisation [documentation](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-configure-auto-features#featurization).

# Unsupervised Learning

Unsupervised Learning:
* train on *unlabeled* data
* attempts to find hidden data structures/common aspects within entities on its own*

Types of unsupervised Learning
* clustering - organize input data into finite number of groups (cluste and tag documents based on the contents of their documents)
* feature learning (representation learning) - transform set of input into other inputs which are potentially more useful in solving a problem, uncovering new and more relevant features for data with multiple categorical features with hight cardinality
* anomality detection, 2 groups
  * normal
  * abnormal

Algorithms examples:
* K-Means clustering
* Principal Component Analysis (PCA)
* Autoencouders

# Semi-supervised learning

Combination of traditional ML supervosed learning and unsupervised learning. 

Obtaining labelled data sets is time-prone and expensive. Obtaining unlabeled data is relatively easy. So most of the time we work with data sets partially labelled.

Types of semi-supervised learning!
* self-learning - obtaining the fully labelled dataset: the model is trained with the labeled data, then used to make predictions for the unlabeled data
* multi-view training - training multiple models on the different views of the data (different feature selection, various model archtectures)
* self-ensemble training - similar to multi-view training, except that the straining is made on a single base model and different hyper-parameters.

## Clustering

Goals of clustering:
* maximazing intra-cluster similarity
* maximizing inter-cluster differences

Applications of clustering:
* personalization and target marketing (goal => develop market campaigns)
* document classification (goal =W tag documents to improve search, create digest or summaries)
* fraud detection (goal => isolate new cases based on historical clusters of fraudulent behaviors)
* medical imaging
* city planning

Clustering algorithms:
* centroid algorithms - is based on the distance of the center of clusters (example: K-means - created up to K clusters and goups similar entities tin hte same cluster)
* density based algorithms - is based on the fact that elements are closely packed together
* distribution based algorithms - is based on the assumption that the data has an inherent distibution ( for example normal distribution), clustering is made base on the probability the member belonning toa particular distribution.
* Hierarchical algorithms - algorithm build a tree of clusters