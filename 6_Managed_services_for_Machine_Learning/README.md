- [Why managed services for ML](#why-managed-services-for-ml)
- [Resources](#resources)
  - [Compute](#compute)
    - [Training compute](#training-compute)
    - [Inferencing compute](#inferencing-compute)
      - [Inferencing in production](#inferencing-in-production)
    - [Actions on Compute](#actions-on-compute)
- [Environments](#environments)
  - [Managed Notebook environments](#managed-notebook-environments)
  - [The Azure Machine Learning SDK](#the-azure-machine-learning-sdk)
- [Basic modelling](#basic-modelling)
  - [Experiment](#experiment)
  - [Runs](#runs)
  - [Models](#models)
  - [Model Registry](#model-registry)
  - [Real-time predictions using Python models - This reference architecture](#real-time-predictions-using-python-models---this-reference-architecture)
- [Resources](#resources-1)
# Why managed services for ML

|| Conventional ML |  Managed services | 
|:-:|:-:|:-:|
|Installation| Installing several applications and libraries on a machine, configuring the environment settings, and then loading any additional resources required to begin working within notebooks or integrated development environments (IDEs). For neural networks, configuring hardware (GPUs)| Very little setup, easy configuration for any needed hardware. Getting the right combination of software versions that are compatible with one another|
| Execution | On local machine | Run from any machine, since everything is in the cloud .|

Examples of computer resources:
* training clusters - training a model
* inferencing clusters - operationalizing the model
* compute instances - contain notebook environment to run notebook based code
* attached compute - using personal VMs attached to ML Azure environments
* local compute - using local computer as a local compute resource

Other services:
* notebooks gallery
* AitoML configurator
* Pipeline designer
* Pipeline manager
* Datasets and dataotre managers
* experiments manager
* models registry
* endpoints manager

# Resources

The compute instance provides a comprehensive set of a capabilities that you can use directly within a python notebook or python code including:

    Creating a Workspace that acts as the root object to organize all artifacts and resources used by Azure Machine Learning.
    Creating Experiments in your Workspace that capture versions of the trained model along with any desired model performance telemetry. Each time you train a model and evaluate its results, you can capture that run (model and telemetry) within an Experiment.
    Creating Compute resources that can be used to scale out model training, so that while your notebook may be running in a lightweight container in Azure Notebooks, your model training can actually occur on a powerful cluster that can provide large amounts of memory, CPU or GPU.
    Using Automated Machine Learning (AutoML) to automatically train multiple versions of a model using a mix of different ways to prepare the data and different algorithms and hyperparameters (algorithm settings) in search of the model that performs best according to a performance metric that you specify.
    Packaging a Docker Image that contains everything your trained model needs for scoring (prediction) in order to run as a web service.
    Deploying your Image to either Azure Kubernetes or Azure Container Instances, effectively hosting the Web Service.


## Compute
A compute target is a designated compute resource or environment where you run training scripts or host your service deployment. 
A managed compute resource is created and managed by Azure Machine Learning. This type of compute is optimized for machine learning workloads. Azure Machine Learning compute clusters and compute instances are the only managed computes. Azure Container Instances are not managed by Azure Machine Learning. 

There are two different variations on compute targets:
* training compute targets 
* inferencing compute targets.

### Training compute

Model training can be performed on the following resources:
*  (large-scale problems)
* compute instances (thought it isn'i its primary purpose)
* local compute

Training clusters can be used for:
* model training
* batch inferences
* generic purposes of running ML Python code

Options:
* Single or multi-node cluster
* Can autoscale when submitting a run
* Clusted and job scheduling are managed automatically
* suport for both CPU and GPU


### Inferencing compute

Types of inferencing:
* real-time - inferences for each new row of data, usually real-time
* batch inference - inferences on multiple rows of data, called  batcn

Inference cluster *is *designed to make real-time inferences*.
For batch inferences, any compute resource can be used.

After the model is ready, you can deploy it:
* web hosted service
* IoT device

Trained models are packaged int containers:

#### Inferencing in production

Compute target for production workloads:
* Azure Kubernetes Service (AKS) - fast response times and auto-scaling for deployed service.
* Azure Machine Learning training cluster - running batch inferencing

Compute targets for specialized deployment:
* Azure Functions
* Azure IoT Edge
* Azure Date Box Edge

### Actions on Compute

Here are the actions on a compute instance:
*  Stop: Since the compute instance runs on a virtual machine (VM), you pay for the instance as long as it is running. Naturally, it needs to run to perform compute tasks, but when you are done using it, be sure to stop it with this option to prevent unnecessary costs.
* Restart: Restarting an instance is sometimes necessary after installing certain libraries or extensions. There may be times, however, when the compute instance stops functioning as expected. When this happens, try restarting it before taking further action.
* Delete: You can create and delete instances as you see fit. The good news is, all notebooks and R scripts are stored in the default storage account of your workspace in Azure file share, within the “User files” directory. This central storage allows all compute instances in the same workspace to access the same files so you don’t lose them when you delete an instance you no longer need.

# Environments

Azure ML environments are an encapsulation of the environment where your machine learning training happens. They define Python packages, environment variables, Docker settings and other attributes in declarative fashion. Environments are versioned: you can update them and retrieve old versions to revisit and review your work.

Environments allow you to:
* Encapsulate dependencies of your training process, such as Python packages and their versions.
* Reproduce the Python environment on your local computer in a remote run on VM or ML Compute cluster
* Reproduce your experimentation environment in production setting.
* Revisit and audit the environment in which an existing model was trained.

Environment, compute target and training script together form *run configuration*: the full specification of training run.

## Managed Notebook environments

Notebook - a combination of runnable code, its output, formatted text and visualizations

Notebook languages:
* Python
* Scala
* R
* SQL
* shell commands

The most popular notebook interfaces:
* Jupiter
* DataBricks notebook
* R-markdown
* Apache Zeppelin

![Performing  five primary stages of model development within a notebook](https://video.udacity-data.com/topher/2020/May/5ebd7e47_managed-notebooks/managed-notebooks.png)

## The Azure Machine Learning SDK
The Azure Machine Learning SDK provides a comprehensive set of a capabilities that you can use directly within a notebook including:

* Creating a Workspace that acts as the root object to organize all artifacts and resources used by Azure Machine Learning.
* Creating Experiments in your Workspace that capture versions of the trained model along with any desired model performance telemetry. Each time you train a model and evaluate its results, you can capture that run (model and telemetry) within an Experiment.
* Creating Compute resources that can be used to scale out model training, so that while your notebook may be running in a lightweight container in Azure Notebooks, your model training can actually occur on a powerful cluster that can provide large amounts of memory, CPU or GPU.
* Using Automated Machine Learning (AutoML) to automatically train multiple versions of a model using a mix of different ways to prepare the data and different algorithms and hyperparameters (algorithm settings) in search of the model that performs best according to a performance metric that you specify.
* Packaging a Docker Image that contains everything your trained model needs for scoring (prediction) in order to run as a web service.
* Deploying your Image to either Azure Kubernetes or Azure Container Instances, effectively hosting the Web Service.
* In Azure Notebooks, all of the libraries needed for Azure Machine Learning are pre-installed. To use them, you just need to import them.

# Basic modelling

In Azure ML an *experiment* is a generic context for handling and organizing runs.
Once an experiment is created, one can create one or more *runs* within it, until she identify the best model. Then the final model is registered in a *model registry*. After registration, one can then *download* or *deploy* the registered model and receive all the files that were registered.

## Experiment

Experiment - a generic *context* or container for handling runs, it is as a logical entity one can use to *organize your model training processes*.

## Runs

One can create model training *runs* within existing experiment.
Run - process of building the trained model. A run contains *all artifacts associated with the training process*: output files, metrics, logs, and a snapshot of the directory that contains scripts.

## Models

The result of a run is a *model*.
A model is a piece of code that takes an input and produces output. To create a model, we start with an  algorithm. By combining this algorithm with the training data and tuned hyperparameters, we produce a more specific function that is optimized for our particular task:
> Model = algorithm + data + hyperparameters

## Model Registry

Azure Machine Learning provides a Model Registry that acts like a version controlled repository for each of your trained models.

Once we have a trained model, we can turn to the *model registry*, which keeps track of all models in an Azure Machine Learning workspace.
Models are either produced by a Run or originate from outside of Azure Machine Learning (and are made available via SDK using model registration).

Model is identified by its name and version. A model can be tagged to be easily searched afterwards. 

## Real-time predictions using Python models - This reference architecture

![Real-time predictions architecture](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/_images/python-model-architecture.png)

The application flow :
1. The trained model is registered to the machine learning model registry.
2. Machine Learning creates a Docker image that includes the model and scoring script.
3. Azure Machine Learning deploys the scoring image on Azure Kubernetes Service (AKS) as a web service.
4. The client sends an HTTP POST request with the encoded request data.
5. The web service created by Azure Machine Learning extracts the data from the request.
6. The data is sent to the Scikit-learn pipeline model for featurization and scoring.
7. The matching results with their scores are returned to the client.



# Resources
* [Course Labs GitHub](https://github.com/solliancenet/udacity-intro-to-ml-labs)
* https://docs.microsoft.com/azure/machine-learning/concept-secure-code-best-practice
* https://notebook78440.eastus2.instances.azureml.net/notebooks/udacity-intro-to-ml-labs/aml-visual-interface/
* [Real-time scoring of Python scikit-learn and deep learning models on Azure](https://docs.microsoft.com/en-us/azure/architecture/reference-architectures/ai/realtime-scoring-python)






