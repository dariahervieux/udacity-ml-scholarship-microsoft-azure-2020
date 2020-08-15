- [Data Import and Transformation](#data-import-and-transformation)
  - [Data wrangling](#data-wrangling)
  - [Managing data in Azure](#managing-data-in-azure)
    - [Datastore](#datastore)
    - [Dataset](#dataset)
      - [Data versioning](#data-versioning)
- [Working with features](#working-with-features)
  - [Feature engineering](#feature-engineering)
  - [Feature selection](#feature-selection)
  - [Data drift](#data-drift)
- [Training model](#training-model)
  - [Parameters vs. Hyperparameters](#parameters-vs-hyperparameters)
  - [Splitting data](#splitting-data)
  - [Training Classifiers](#training-classifiers)
  - [Training Regressors](#training-regressors)
  - [Evaluating performance](#evaluating-performance)
    - [Evaluation mertics for classification](#evaluation-mertics-for-classification)
      - [Confusion matrices](#confusion-matrices)
    - [Evaluation metrics for regression](#evaluation-metrics-for-regression)
  - [Training multiple models](#training-multiple-models)
    - [ensemble training](#ensemble-training)
    - [Automated training on Azure Ml](#automated-training-on-azure-ml)
- [Resources](#resources)

# Data Import and Transformation
Most algorythm are secitive to the quality of data they are learning from. That's why data transformation is a *crucial* step in getting a high quality model.
Problems in data: noise, wrong values (outliers), missing values.

## Data wrangling 
Data wrangling is the process of *cleaning and transforming* data to make it more appropriate for data analysis. The main steps:
* Understand the data: explore the raw data and check the general quality of the dataset.
* Transform the raw data: restructuring, normalizing, cleaning. For example, this could involve handling *missing values* and *detecting errors*.
* Make the data available: Validate and publish the data.

## Managing data in Azure

![Datastore and Datasets (source Microsoft)](https://docs.microsoft.com/en-us/azure/machine-learning/media/concept-data/data-concept-diagram.svg)

The steps of the data access workflow:
1. Create a Datastore so that you can access storage services in Azure.
2. Create a Dataset, which you will subsequently use for model training in your machine learning experiment.
3. Create a dataset Monitor to detect *issues* in the data, such as data drift.

Data drift - the data which changes/evolves overtime which makes models trained on initial data less performant.

### Datastore

Datastore is a *layer of abstraction* over the supported Azure storage services. 
* connection information is hidden which provides a layer of security
* answers to the question: "How do a I securely connect to my data in my Azure Storage?".
* compute location independence
* every ML service comes with pre-configured default Datastore
*  Azure Blob container and blob data store are configured to serve as default Datastores

### Dataset

Dataset is a resource: a set of concrete files for exploring, transforming, testing, validating, etc. data.
* reference that points to the data in your Datastore, thus a dataset can be *shared* across other experiments without copying the data.
* answers to the questions:
  * "How do a I get specific data files in my Datastore?".
  * "How do I get naccess to specific data files?".
* can be created from local files, public URLs, Azure Open Datasets, and files uploaded to the datastores.
* used to interact with your data in the datastore and to package data into consumable objects. 
* can be versioned

Datasets allow:
* Have a single copy of some data in your storage, but reference it multiple times—so that you don't need to create multiple copies each time you need that data available.
* Access data during model training without specifying connection strings or data paths. 
* More easily share data and collaborate with other users, use the data for different experiments.
* Bookmark the state of your data by using *dataset versioning*.

Dataset types supported in Azure ML Workspace:
* The *Tabular Dataset*, which represents data in a tabular format created by parsing the provided file or list of files.
* The *Web URL* (File Dataset), which references single or multiple files in datastores or from public URLs.


#### Data versioning

Versionning gives access to traceability. 

When to do versioning:
* New data is available for retraining.
* When you are applying different approaches to data preparation or feature engineering.

# Working with features

## Feature engineering

Sometimes existing features are not enough to create high quality models.
In this case we can create new features, which are derived from the original data:
* mathematically (using SQL, programming)
* by applying a ML algorithm

Classical ML algorithms depend much more on feature engineering than deep learning ones.


Feature engineering tasks on numerical data:
* aggregation (sum, count, mean, ..)
* part of (part of the date: time, year, ..)
* binning ( group data items in bins and apply aggregation on each of these beans)
* flagging - deriving boolean values
* frequency based - calculate frequency
* embedding (feature learning)
* derive by example - lean new features using examples of existing features

Text data must be first translated into numeric form.
Text embedding  (transformation into numeric form) approaches:
* text frequency
* inverse document frequency
* word embedding

Image data is also translated into numeric form. And the form of the representation (matrix of RGB tuples) can be different: array of w*h*3 length, or 3D array,..

## Feature selection

Feature selection - selecting the features that are most important or most relevant for a given model.
Even if there is no feature engineering involved, the original data set can have a huge amount of feature.

Reasons for feature selection:
* eliminate irrelevant, redundant or highly correlated features.
* reduce dimensionality

The *curse of dimensionality* - many machine learning algorithms cannot accommodate a large number of features, so it is often necessary to do dimensionality reduction to decrease the number of features.

Algorythms to reduce dimensionality:
* [PCA](https://en.wikipedia.org/wiki/Principal_component_analysis) - mathematical approach based mostly on exact mathematical calculations
* [t-SNE (t-Distributed Stochastic Neighboring Entities)](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) - probabilistic approach; useful for the visualization of multidimensional data.
* Feature embedding - encodes a larger number of features into a smaller number of "super-features."

Azure ML prebuilt modules for feature selection:
* Filter-based feature selection: identify columns in the input dataset that have the greatest predictive power
* Permutation feature importance: determine the best features to use by computing the feature importance scores

## Data drift

Data drift - change in data which leads to model's performance degradation, the top reason for model degradation.

Most common causes:
* changes in upstream process (A sensor is replaced, causing the units of measurement to change)
* data quality issues (a sensor outage)
* data natural evolution (for example customer behavior changes over times)
* change in feature correlation

Data drift must be monitored and detected as early as possible to retrain the model.

The process of monitoring for data drift involves:
* Specifying a baseline dataset – usually the training dataset
* Specifying a target dataset – usually the input data for the model
* Comparing these two datasets over time, to monitor for differences


Scenarios for setting up for data monitoring:
1. detecting data drift - detect difference between the model input vs. training data
2. detecting data drift in timeseries dataset - detecting season data drift : current input data set vs. previous period input data set
3. performing analysis on past data

Understanding data drift results in Azure:
* data drift magnitude - the percentage of data similarity: 0% - identical, 100% - completely different data set
* drift contribution by feature - the contribution of each feature in a drift, helps to identify drifting features

# Training model

## Parameters vs. Hyperparameters

A major goal of model training is to learn the values of the *model parameters*.
In contrast, some model parameters are not learned from the data. These are called *hyperparameters* and their values are set before training. When choosing hyperparameters values, we make the *best guess*, and then we *adjust them* based on the model's performance. 

Examples of hyperparameters:
* The number of layers in a deep neural network;
* The number of clusters (such as in a k-means clustering algorithm);
* The learning rate of the model.

## Splitting data

Splitting the Data

Datasets:
1. Training data - to learn the values for the *parameters*
2. Validation data - to check the model's performance and to tune the *hyperparameters*(run on training data) until the model performs well with the validation data
3. Test data - to do a final check of model's performance

## Training Classifiers

In a classification problem, the outputs are *categorical* or *discrete*.

Three main types of classification problem:
* binary classification (classify medical test results as "positive" or "negative", fraud detection) - the most difficult problem are often in this category,
* multi-class single-label classification (classify an image as one (and only one) of five possible fruits, recognition of numbers)
* multi-class multi-label classification (text tagging, classify music as belonging to multiple groups (e.g., "upbeat", "jazzy", "pop").)

## Training Regressors

In a regression problem, the output is *numerical* or *continuous*.

Two main types of regression problems:
* regression to arbitrary values - no boundary defined (price of houses based on different inputs)
* regression to values between 0 and 1 (the probability of a certain transaction to be fraudulent)

Examples og algorithms:
* linear regressor
* decision forest regressor

## Evaluating performance

Performance is evaluated on the test data. The test dataset is a portion of labeled data that is split off and reserved for model evaluation.

When splitting the available data, it is important to preserve the *statistical properties* of that data. This means that the data in the training, validation, and test datasets need to have *similar statistical properties* as the original data to prevent bias in the trained model.

Metrics used for evaluation depend on the problem we need to solve.
Establishing a set of criterias for evaluation is important:
* what is the primary metric used for evaluation?
* what us the threshold of that metric that we want to meet or to exceed to have "a good enough" model

### Evaluation mertics for classification

#### Confusion matrices

|| Actual Class X | Actual Class Y| 
|:-|:-|:-|
|Predicted class X positive| correct X - True Positive (TP)| wrong X - False positive (FP) |
|Predicted class Y negative| incorrect y - False Negavites (FN)|  correct Y - True Negative (TN)|

Evaluation metrics:
* accuracy - the proportion of correct predictions:
> (TP + TN) / (TP + FP + FN + TN​)

* precision / - proportion of positive cases that were correctly identified; **; How many selected items are relevant?
> TP / (TP + FP)

* recall / *sensitivity* - proportion af actual positive cases that were identified; *true positive rate*; How many relevant items are selected?
> TP / (TP + FN)

* false positive rate (close to type I error)
> FP / (FP + TN)

* *specitivity* - How many negative selected elements are trully negative.

* F1 - mesures the balance between precision and recalls
> 2∗Precision∗Recall / (Precision+Recall)​

To fully evaluate the effectiveness of a model, one must examine both precision and recall. Since precision and recall are often in tension: improving precision typically reduces recall and vice versa.
F1  is a measure of a test's accuracy. It takes into consideration both the precision and recall. F1 reaches its best value at 1, which means that precision and recall have perfect values (perfect balance between precision and recall).
Precision = What proportion of positive identifications was actually correct? =  TP / (TP + FP)
Recall = What proportion of actual positives was identified correctly? =  TP / (TP + FN)
Precision = What proportion of positive identifications was actually correct? =  TP / (TP + FP)
Recall = What proportion of actual positives was identified correctly? =  TP / (TP + FN)

Charts:
* Receiver Operating Characteristics (ROC) chart - the chart of *false positive rate* against the rate of FP
  * Area Under the Curve (AUC)
    * for random classifier the value is 0.5 (diagonal)
    * for perfect classifier (100% TP) the area is 1
  
* Gain and lift chart - measure how much better one can expect to do with the predictive model comparing without a model

### Evaluation metrics for regression

Metrics:
* Root Mean Squared Error (RMSE) - the square root of the average of the squared differences between the predictions and the true values.
* Mean Absolute Error (MAE) - Average of the absolute difference between each prediction and the true value.
* R-squared metrics (in statistics - the coefficient of determination) - How close the regression line is to the true values.
* Spearman rank correlation - technique to measure the strength and the direction of the correlation between predicted values and the true values

Charts:
* Predicted vs true chart
  * ideal regressor
  * average predicted values
  * histogram of distribution of true values in predicted result (should have bell-shape)
* Histogram of residuals (true value - predicted value) - must be normally distributes (bell shape)

## Training multiple models

Strength in numbers principle - to reduce the error of an individual trined ML model, we can train several models and combine their resuts.

### ensemble training

Ensemble learning  - a technique to combine multiple machine learning models to produce one predictive model. 
Individual algorithm trains several models as its internal training process.

Types of ensemble algorithms:
* Bagging/bootstrap aggregation
  * uses *randomly selected subsets* of data to train several homogeneous models (bag of models)
  * the final prediction is an equally weighted average prediction from individual models
  * helps to reduce *overfitting* models that tend to have high variance (such as decision trees)
* Boosting
  * uses the *same input data* to train multiple models using *different hyperparameters*.
  * sequential training,  with each new model correcting errors from previous models
  * the final predictions are a weighted average from the individual models
  * helps reduce *bias* for models.
* Stacking
  * trains a large number of completely different (heterogeneous) models
  * combines the outputs of the individual models into a meta-model that yields more accurate predictions


### Automated training on Azure Ml

We want to scale up the process of training the models. We have many models trained separately and we combine their results to get better outputs. Automated training is a way to get a baseline to create a model with a decent performance.

Automated machine learning automates many of the iterative, time-consuming, tasks involved in model development: selecting the best features, scaling features optimally, choosing the best algorithms, tuning hyperparameters.
Inputs to Automated training in Azure:
* dataset
* compute resources setup
* select a type of task to perform (classification, regression, forecasting)
* select primary evaluation metric
* critaria for finishing the process/ level of a quality for the model

# Resources
* [Secure data access in Azure Machine Learning](https://docs.microsoft.com/en-us/azure/machine-learning/concept-data)
* [What is an Azure Machine Learning workspace?](https://docs.microsoft.com/en-us/azure/machine-learning/concept-workspace)
* https://www.listendata.com/2014/08/excel-template-gain-and-lift-charts.html

