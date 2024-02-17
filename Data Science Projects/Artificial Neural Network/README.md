
# Bank Customer Churn Prediction using ANN

## Project Description

This project aims to predict the likelihood of a bank's customers leaving the bank (churn rate) using Artificial Neural Networks (ANN). By analyzing customer data and behaviors, we can identify patterns that contribute to customer attrition. The ANN model is trained to recognize these patterns and predict future churn, enabling the bank to implement retention strategies more effectively.

## Data Description

The dataset used for this project comprises customer information from a banking institution. It includes details such as customer ID, credit score, geographical location, gender, age, tenure, account balance, number of products used, credit card status, active membership status, estimated salary, and churn label.
| Feature          	| Description                                                                     	|
|------------------	|---------------------------------------------------------------------------------	|
| Customer ID      	| Unique identifier for each customer                                             	|
| Surname          	| Customer's surname                                                              	|
| Credit Score     	| Customer's credit score                                                         	|
| Geography        	| Customer's country of residence                                                 	|
| Gender           	| Customer's gender                                                               	|
| Age              	| Customer's age                                                                  	|
| Tenure           	| Number of years the customer has been with the bank                             	|
| Balance          	| Current balance in the customer's account                                       	|
| NumOfProducts    	| Number of products the customer has with the bank                               	|
| HasCrCard        	| Whether the customer has a credit card with the bank (Yes/No)                   	|
| IsActiveMember   	| Whether the customer is an active member of the bank's loyalty program (Yes/No) 	|
| Estimated Income 	| Estimated annual income of the customer                                         	|
| Exited           	| Whether the customer has left the bank (Yes/No)                                 	|

### Pre-processing Steps

- Encoding categorical variables: Geography, Exited and Gender have been encoded into numerical values.
- Feature Scaling: All numerical values have been scaled to ensure uniformity for the ANN.
- Handling Missing Values: Rows with missing data have been appropriately handled either by removal or by imputation.

## Model Architecture

The ANN model is designed with:

- Input Layer: Consists of units equal to the number of features in the dataset (excluding the target variable 'Exited').
- Hidden Layers: Two hidden layers with ReLU activation function to introduce non-linearity.
- Output Layer: A single neuron with a sigmoid activation function to predict the churn probability.
The model uses the Adam optimizer and binary cross-entropy loss function, which is suitable for binary classification problems.

## Training and Evaluation

The model is trained with a batch size of 32 and for 100 epochs, using an 80-20 train-test split of the dataset.

### Metrics Used for Evaluation:
- Accuracy: The proportion of correctly predicted observations to the total observations.
- Precision: The ratio of correctly predicted positive observations to the total predicted positive observations.

## Results and Insights

The trained ANN model yielded an accuracy of 97%, and a precision of 94%. Key insights drawn from the model include:

- The importance of features like geography, age, and balance in predicting customer churn.
- Visualizations of loss and accuracy over epochs, showing model convergence.

## Code

The code for this project is available in the form of Jupyter Notebooks, which include detailed comments explaining each step of the process, from data pre-processing to model evaluation.

Feel free to explore the notebooks and use them as a foundation for your own predictive modeling projects.

