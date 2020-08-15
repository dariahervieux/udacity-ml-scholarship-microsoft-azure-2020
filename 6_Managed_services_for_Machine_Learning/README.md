- [Why managed services for ML](#why-managed-services-for-ml)
- [Resources](#resources)
  - [Compute](#compute)
  - [Training compute](#training-compute)
  - [Inferencing compute](#inferencing-compute)
    - [Inferencing in production](#inferencing-in-production)
  - [Actions on Compute](#actions-on-compute)
  - [Managed Notebook environments](#managed-notebook-environments)
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

## Compute
A compute target is a designated compute resource or environment where you run training scripts or host your service deployment. 
A managed compute resource is created and managed by Azure Machine Learning. This type of compute is optimized for machine learning workloads. Azure Machine Learning compute clusters and compute instances are the only managed computes. Azure Container Instances are not managed by Azure Machine Learning. 

There are two different variations on compute targets:
* training compute targets 
* inferencing compute targets.

## Training compute

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


## Inferencing compute

Types of inferencing:
* real-time - inferences for each new row of data, usually real-time
* batch inference - inferences on multiple rows of data, called  batcn

Inference cluster *is *designed to make real-time inferences*.
For batch inferences, any compute resource can be used.

After the model is ready, you can deploy it:
* web hosted service
* IoT device

Trained models are packaged int containers:

### Inferencing in production

Compute target for production workloads:
* Azure Kubernetes Service (AKS) - fast response times and auto-scaling for deployed service.
* Azure Machine Learning training cluster - running batch inferencing

Compute targets for specialized deployment:
* Azure Functions
* Azure IoT Edge
* Azure Date Box Edge

## Actions on Compute

Here are the actions on a compute instance:
*  Stop: Since the compute instance runs on a virtual machine (VM), you pay for the instance as long as it is running. Naturally, it needs to run to perform compute tasks, but when you are done using it, be sure to stop it with this option to prevent unnecessary costs.
* Restart: Restarting an instance is sometimes necessary after installing certain libraries or extensions. There may be times, however, when the compute instance stops functioning as expected. When this happens, try restarting it before taking further action.
* Delete: You can create and delete instances as you see fit. The good news is, all notebooks and R scripts are stored in the default storage account of your workspace in Azure file share, within the “User files” directory. This central storage allows all compute instances in the same workspace to access the same files so you don’t lose them when you delete an instance you no longer need.


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

# Resources
https://docs.microsoft.com/azure/machine-learning/concept-secure-code-best-practice
https://notebook78440.eastus2.instances.azureml.net/notebooks/udacity-intro-to-ml-labs/aml-visual-interface/lab-19/notebook/1st-experiment-sdk-train-model.ipynb






