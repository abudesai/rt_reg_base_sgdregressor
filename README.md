SGD Regression in Scikit-Learn for Regression

- stochastic gradient descent
- scikit learn
- regularization
- python
- feature engine
- scikit optimize
- flask
- nginx
- gunicorn
- docker
- abalone
- auto prices
- computer activity
- heart disease
- white wine quality
- ailerons

This is an SGD (Stochastic Gradient Descent) Regressor implemented using Scikit-Learn.

SGD is an optimization technique with a general approach to minimize the cost function. In Stochastic Gradient Descent, a few samples are selected randomly instead of the whole data set for each iteration. SGD is computationally less expensive any other typical Gradient Descent algorithms.

The class SGDRegressor in Scikit-Learn implements a plain stochastic gradient descent learning routine which supports different loss functions and penalties to fit linear regression models.

The data preprocessing step includes:

- for categorical variables
  - Handle missing values in categorical:
    - When missing values are frequent, then impute with 'missing' label
    - When missing values are rare, then impute with most frequent
- Group rare labels to reduce number of categories
- One hot encode categorical variables

- for numerical variables

  - Add binary column to represent 'missing' flag for missing values
  - Impute missing values with mean of non-missing
  - MinMax scale variables prior to yeo-johnson transformation
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale data after yeo-johnson

- for target variable
  - Use Yeo-Johnson transformation to get (close to) gaussian dist.
  - Standard scale target data after yeo-johnson

HPT includes choosing the optimal values for alpha, l1_ratio and stopping criterion.

During the model development process, the algorithm was trained and evaluated on a variety of datasets such as abalone, auto_prices, computer_activity, heart_disease, white_wine, and ailerons.

The main programming language is Python. Other tools include Scikit-Learn for main algorithm, feature-engine and Scikit-Learn for preprocessing, Scikit-Learn for calculating model metrics, Scikit-Optimize for HPT, Flask + Nginx + gunicorn for web service. The web service provides two endpoints- /ping for health check and /infer for predictions in real time.
