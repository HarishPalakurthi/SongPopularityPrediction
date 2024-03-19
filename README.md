# SongPopularityPrediction

This project aims to predict the popularity of songs based on various features such as artists, country, explicit content, and audio features like danceability, energy, and tempo. The predictions are made using machine learning models, including linear regression and neural networks.

## Data Transformation

The data transformation process involves preparing the dataset for modeling by performing various operations such as:

- **Label Encoding**: Categorical variables like artists and country are encoded into numerical values using label encoding to facilitate model training.

- **Standardization**: Numerical features are standardized to ensure consistency in scale across different variables. This helps improve the performance of machine learning models that are sensitive to the scale of input features.

- **Handling Missing Values**: Missing values are addressed by either removing records with missing values or imputing them with appropriate values to ensure the integrity of the dataset.

## Modeling

The modeling phase consists of building and training machine learning models to predict song popularity based on the transformed dataset. The models used in this project include:

- **Linear Regression**: A simple linear regression model trained on numerical features to predict song popularity.

- **Neural Networks**: Neural network models with multiple hidden layers trained on both categorical and numerical features to capture complex relationships between input features and the target variable. These models utilize activation functions like ReLU and linear activation for the output layer.

The performance of each model is evaluated using metrics such as Mean Squared Error (MSE) and R-squared to assess their effectiveness in predicting song popularity accurately. Additionally, the models are trained, validated, and tested to ensure robustness and generalization to unseen data.
