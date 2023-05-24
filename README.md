# Deep Learning Challenge

# Project Description 

## Step 1: Preprocess the Data
- Read in the charity_data.csb to a Pandas DataFrame and identify the targets and features for the model. 
- Drop the EIN and NAME columns.
- Determine the number of unique values for each column. 
- For columns that have more than 10 unique values, determine the number of data points for each unique value.
- Use the number of data points of each unique value to pick a cutoff value and bin categorical variables into a new value "Other".
- Use pd.get_dummies() to encode categorical variables. 
- Split the preprocessed data into a features and target array and use the arrays and train_test_split to split the data into training and testing datasets. 
- Scale the training and testing features using StandardScaler, fit it to the training data, and use transform. 

## Step 2: Compile, Train, and Evaluate the Model
- Use the notebook file in Google Colab and create a neural network by assigning the number of input features and nodes for each layer using TensorFlow and Keras.
- Create the first hidden layer and choose an appropriate activation function. 
- Add a second hidden layer with an appropriate activation function if needed. 
- Check the structure of the model.
- Compile and train the model.
- Create a callback that saves the model's weights every five epochs.
- Evaluate the model using the test data to determine the loss and accuracy. 
- Save and export your results to an HDF5 file named AlphabetSoupCharity.h5

## Step 3: Optimize the Model
- Optimize the model to achieve a target predictive accuracy higher than 75% by dropping more or fewer columns, creating more bins for rare occurrences in columns, increasing or decreasing the number of values for each bin, add more neurons to a hidden layer, add more hidden layers, use different activation functions for the hidden layers, or add or reduce the number of epochs to the training regimen. 
- Create a Google Colab file and name it AlphabetSoupCharity_Optimization.ipynb
- Import dependencies and read in the charity_date.csv to a DataFrame. 
- Preprocess the dataset and adjust for any modifications that came out of optimizing the model.
- Design a neural network model and adjust for modifications that will optimize the model to achieve higher than 75% accuracy. 
- Save and export your results to an HDF5 file named AlphabetSoupCharity_Optimization.h5.

## Step 4: Write a Report on the Neural Network Model

### Overview of the analysis: 
The purpose of this analysis is to use deep learning techniques to predict if applicants for the nonprofit foundation Alphabet Soup would be successful or not.

### Results:
- Data Preprocessing
  - What variable(s) are the target(s) for your model?
  - The target variable for this model is the "IS_SUCCESSFUL" variable.
  - What variable(s) are the features for your model?
  - The features for this model are the "ORGANIZATION", "ASK_AMT", "SPECIAL_CONSIDERATIONS", "INCOME_AMT", "STATUS", "USE_CASE", "CLASSIFICATION", "AFFILIATION", and "APPLICATION_TYPE" variables. 
  - What variable(s) should be removed from the input data because they are neither targets nor features?
  - The variables that should be removed are the "EIN" and "NAME" variables. 
- Compiling, Training, and Evaluating the Model
  - How many neurons, layers, and activation functions did you select for your neural network model, and why?
  - For the original model, I used two hidden layers with one output layer, 80 nodes in the first layer and 30 in the second, and relu for the hidden layer activation functions and sigmoid for the output layer. ***
  - Were you able to achieve the target model performance?
  - I was not able to achieve the target model performance for this initial model (72.5% accuracy). With the three other optimization attempts, I was still not able to achieve the target model performance (). 
  - What steps did you take in your attempts to increase model performance? 
  - For the first optimization attempt, I dropped the organization column and changed the threshold for the others category to greater than 50 for application type and classification. For the second optimization attempt, I added more neurons (100 and 60) to the two hidden layers and added an additional hidden layer with 20 neurons. For the last attempt, I changed the activation function to sigmoid for all the layers and increased the epochs to 200. 
### Summary:

## Step 5: Copy Files into your Repository
- Download your Colab notebooks to your computer.
- Move them into your Deep Learning Challenge directory in your local repository.
- Push the added files to GitHub.

# Software and Files
- from sklearn.model_selection import train_test_split
- from sklearn.preprocessing import StandardScaler
- import pandas as pd
- import tensorflow as tf
- import pandas as pd 
- CSV: https://static.bc-edx.com/data/dl-1-2/m21/lms/starter/charity_data.csv

# Output/Analyses

## Step 1: Preprocess the Data
- Created a DataFrame containing the charity_data.csv and identified the target and feature dataset.
<img width="1439" alt="Screenshot 2023-05-24 at 3 01 38 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/ad89cbcc-cdae-462e-b5eb-12f55112ae0e">
- Dropped the EIN and NAME columns.
<img width="1444" alt="Screenshot 2023-05-24 at 3 02 05 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/a6d93eb6-d1e4-44b2-8fbf-bfc20675443d">
- Determined the number of unique values in each column.
<img width="1447" alt="Screenshot 2023-05-24 at 3 02 29 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/e76a16d1-d4b0-49eb-8790-13f9ca722951">
- For columns with more than 10 unique values, determined the number of data points for each unique value. 
<img width="1436" alt="Screenshot 2023-05-24 at 3 02 52 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/41759676-edd8-44c3-a02e-edef93647073">
<img width="1441" alt="Screenshot 2023-05-24 at 3 03 40 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/fc2ad64c-88b8-40ce-b835-9cfddb766b50">
- Created a new value called Other that contains rare categorical variables. 
<img width="1442" alt="Screenshot 2023-05-24 at 3 03 14 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/490e27a5-6154-4e2d-abdb-dc68162540d5">
<img width="1442" alt="Screenshot 2023-05-24 at 3 03 55 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/c5ae8f44-f973-4f3e-981e-0cc2eba56d7d">
- Converted categorical data to numeric.
<img width="1447" alt="Screenshot 2023-05-24 at 3 04 41 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/cfb256cd-fd31-4f79-b383-648a550e01f9">
- Created a feature array, X and a target array y by using the preprocessed data.
<img width="1138" alt="Screenshot 2023-05-24 at 3 05 15 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/8c76b0ba-14b3-459b-9e9a-ddc110133526">
- Split the preprocessed data into training and testing datasets.
<img width="1100" alt="Screenshot 2023-05-24 at 3 05 42 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/e93797e4-5456-4017-b7fe-8a8327c0c87b">
- Scaled the data using a StandardScaler that has been fitted to the training data. 
<img width="1276" alt="Screenshot 2023-05-24 at 3 06 05 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/cf732903-42c7-4a3e-a58b-41ef16f9d37e">

## Step 2: Compile, Train, and Evaluate the Model
- Created a neural network model with a defined number of input features and nodes for each layer.
<img width="1113" alt="Screenshot 2023-05-24 at 3 11 29 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/dc040aa8-7b26-4eec-86b8-d5f5883397e5"
- Created hidden layers and an output layer with appropriate activation functions.
<img width="1111" alt="Screenshot 2023-05-24 at 3 12 35 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/251e152c-ec50-45df-ae87-335568603fd1">
- Checked the structure of the model.
<img width="1018" alt="Screenshot 2023-05-24 at 3 12 56 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/2bc47e60-1efc-4ede-8086-3c03d3eba754">
- Compiled and trained the model.
- <img width="1172" alt="Screenshot 2023-05-24 at 3 13 11 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/ad5a21c4-c45d-4c24-aaf5-460bdd1873fd">
- Evaluated the model using the test data to determine loss and accuracy. 
<img width="1179" alt="Screenshot 2023-05-24 at 3 13 30 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/5633cf2f-6103-42ca-8403-bff07a26da35">
- Exported the results to an HDF5 file named AlphabetSoupCharity.h5.
<img width="664" alt="Screenshot 2023-05-24 at 3 13 43 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/99af4b01-59f0-4a0e-aaf6-c1a8222c889a">

## Step 3: Optimize the Model
- Repeated the preprocessing steps in a new Jupyter notebook. 
<img width="1177" alt="Screenshot 2023-05-24 at 3 16 07 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/5ec8e56b-5da6-4c92-8acf-ac2154ee0c38">
- Created a new neural network model, implementing at least 3 model optimization methods.
- Optimization Attempt 1: Dropped an additional column (Organization) and created more bins for rare occurrences in columns by changing the threshold for others category <50 for both the application type and classification columns.
<img width="1358" alt="Screenshot 2023-05-24 at 3 18 46 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/17247f15-e193-4008-855c-10541846b7d3">
<img width="1375" alt="Screenshot 2023-05-24 at 3 19 02 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/c447bcfa-a6b5-43cf-8cad-cb73d847a42d">
<img width="1367" alt="Screenshot 2023-05-24 at 3 19 18 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/b343fb72-3b8b-4227-b6a7-cdfc0efa9f08">
- Optimization Attempt 2: Added more neurons to hidden layers (the first and second layer by 100 and 60, respectively) and added an additional hidden nodes layer.
<img width="1353" alt="Screenshot 2023-05-24 at 3 20 39 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/9f9af80c-6c0d-4aa3-9fdf-479d7c9c5d0a">
- Optimization Attempt 3: Used the sigmoid acitvation function for all the layers and increased the number of epochs to 200. 
<img width="1338" alt="Screenshot 2023-05-24 at 3 21 32 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/9c10dc7d-36ad-46a7-bbf0-ecb3bde793f0">
- Saved and exported the results to an HDF5 file named AlphabetSoupCharity_Optimization.h5.
<img width="1190" alt="Screenshot 2023-05-24 at 3 21 54 PM" src="https://github.com/brittanynicole7/deep-learning-challenge/assets/119909433/877f8046-d846-463c-97e5-e73dc06209d5">

## Step 4: Addressed questions regarding the purpose of the analysis, the results, and an overall summary of the process (see above). 

# Author 
-Brittany Wright github:brittanynicole7
