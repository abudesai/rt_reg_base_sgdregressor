SGD Regression in Scikit-Learn for Regression

* stochastic gradient descent
* scikit learn
* regularization
* python
* feature engine
* scikit optimize
* flask
* nginx
* gunicorn
* docker
* abalone
* auto prices
* computer activity
* heart disease
* white wine quality
* ailerons

This is an SGD (Stochastic Gradient Descent) Regressor implemented using Scikit-Learn. 

SGD is an optimization technique with a general approach to minimize the cost function. In Stochastic Gradient Descent, a few samples are selected randomly instead of the whole data set for each iteration. SGD is computationally less expensive any other typical Gradient Descent algorithms. 

The class SGDRegressor in Scikit-Learn implements a plain stochastic gradient descent learning routine which supports different loss functions and penalties to fit linear regression models.

Preprocessing includes missing data imputation, standardization, one-hot encoding etc. For numerical variables, missing values are imputed with the mean and a binary column is added to represent 'missing' flag for missing values. For categorical variable missing values are handled using two ways: when missing values are frequent, impute them with 'missing' label and when missing values are rare, impute them with the most frequent. 

HPT includes choosing the optimal values for alpha,  l1_ratio and stopping criterion. 

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, feature-engine and Scikit-Learn for preprocessing, Scikit-Learn for calculating model metrics, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides two endpoints- /ping for health check and /infer for predictions in real time. 