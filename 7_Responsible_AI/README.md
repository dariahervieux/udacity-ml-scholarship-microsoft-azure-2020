
# PARFIT Microsoft AI Principles

*Privacy* - AI systems should protect people and their personal data.
*Accountability* - Algorithms and the people who write them should be responsible or answerable for their impacts.
*Reliability and Safety* - AI systems should perform consistently and minimize risk.
*Fairness* - AI systems should treat all people fairly.
*Inclusiveness* - AI systems should empower everyone and engage people
*Transparency* - AI systems should be understandable

Foundational principles that ensure the effectiveness of the other principles: *Accountability* and *Transparency*.

# Model transparency and explainabiliy

One of the challenges of ML is the ability to examine and explain the inner logic of a model, in order to understand how the results of the model were calculated.
Algorithmic models, like for example Decision Trees are very easy to understand and to explain.
Neural networks, on the other hand, are the most opaque models which are really hard to explain.

Feature Importance is the most important aspect of expandability, it tells which features contribute most to the resulting predictions.

## Azure ML Explainers
Azure ML proposes several types of [explainers](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability#supported-interpretability-techniques).

Types:
* Direct explainer - selected manually based on the model type when one knows which explainer will work best,
* Meta-explainers -  used for the automatic selection of *direct explainers*. Based on a given model and a dataset, a meta explainer will select the best direct explainer and use it to generate the explanation information.

### Direct explainers
Direct explainers can be:
* model agnostic - can be used to explain any model regardless of the algorithm that was used to train it (Mimic explainer, SHAP Kernel explainer),
* model specific - depends on the algorithm used for the model (SHAP Tree exapliner, SHAP Deep Explainer).
* 
Mimic direct explainer can be used for any model, it created its own model that is trained to approximate the predictions of the original model, this approximated model can be explained.

### Meta explainers
Meta explainers are named after the *type of data the model uses*, whereas direct explainers are named according to the approach they take. Meta explainers vary based on data type: Tabular explainer, Text explainer, Image Explainer.

TabularExplainer meta-explainer uses one of three explainers: TreeExplainer, DeepExplainer, or KernelExplainer, and is automatically selecting the most appropriate one for our use case.

### Explaining models

All explainers are available in Azure ML SDK. One can use SDK's API in a notebook to make global or local explanations and explore the graphical result. Result explanation is also available in AzureML Studio.

## Global Explanation Using a Meta Explainer (TabularExplainer)

Global Model Explanation is a holistic understanding of how the model makes decisions. It provides you with insights on what features are most important and their relative strengths in making model predictions.

# Model fairness

[Fairlearn](https://fairlearn.github.io/) is a toolkit to identify and mitigate unfairness in machine learning models. It contains algorithm for mitigating unfairness in binary classification and regression.

Fairlearn follow the approach of *group fairness*: "Which groups of individuals are at risk of experiencing harms?". The relevant groups need to specified by the data scientists. So a data scientists selects "sensitive" features, for example "sex".
Failearn trains several models (via GridSearch) which try to mitigate the unfairness detected.

Demos notebooks demonstrating Fairlearn capabilities is available in [Fairelearn GitHub repo](https://github.com/fairlearn/fairlearn/tree/master/notebooks).

# Resources

* https://docs.microsoft.com/en-us/azure/machine-learning/how-to-machine-learning-interpretability
* https://docs.microsoft.com/en-us/azure/machine-learning/
* [Fairlearn gitHub repo](https://github.com/fairlearn/fairlearn)