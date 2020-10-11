# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Summary
In this problem we have been given a dataset of bankmarketing. In that there is a demographic profile of a customer and we want topredict whether we should give the loan to the customer.
We can use his demographic profile as independent variablesto predict whether that customer is likely to default.
We used hyperdrive model (hyperparametr tuning) and an automl model for this purpose.

## Scikit-learn Pipeline
**Explain the pipeline architecture, including data, hyperparameter tuning, and classification algorithm.**
Pipeline architecture is:
1. Dataset ingestion. The data was related to housing loans and each observation represents a customer profile.
2. Split the data intotrain and test so that we select themodel based on test set performance.
3. Then we create an estimator config,in our case the estimator is SKLearn.
4. As we need to clasify the customer profile, we used logistic regression to serve our purpose.
5. We create configurations for each of hyperdrive and automl.
6. We give estimator configuration as input of hyperdrive configuration. Automl doesn't require this.
7. We then run both the experiments andcompare their performance one on one.

**What are the benefits of the parameter sampler you chose?**
Here we are given random sampler. We can choose grid sampler for better performance if we know the tentative range under which modelbehaves as expected.
We can use random sampler first to come up with the vague range and then fine tune it further using grid or Bayesian sampler.
Bayesian will always improve the primary metric by comparingit with the metric of the previous one.

**What are the benefits of the early stopping policy you chose?**
We chose slack factor as 0.1 which gives less allowance to make any errors. At the same timewe gave evaluation interval as 4 so that after each 4 runs the policy would evaluate which runs to terminate.

## AutoML
AutoML believes in C and max_iter parameters as respectively.

## Pipeline comparison
Both the automl and hyperdrive architectures are similar however there are certain defining differences.
For example, hyperdrive offers us a greater degree of flexibility and control. We can specify the range, distribution, termination policies to govern the overall modelrun.
If we are subject matter experts, then probably it is better to run hyperdrive than automl.
In this scenario, automl gave us the best results as compared with that of hyperdrive.

## Future work
1. We can have more total number of runs, more powerful compute cluster
2. We can builda pipeline where it first runs automl and themwe can fine tune it using hyperdrive.
3. We can use other algorithms like neural network instead of conventional logistic regression.
4. There's a class imbalance detected. We can use SMOTE to balance the input data. Then run the same experiment. 

## Proof of cluster clean up
**If you did not delete your compute cluster in the code, please complete this section. Otherwise, delete this section.**
**Image of cluster marked for deletion**
![Alt text](./image/cluster_delete.png)
