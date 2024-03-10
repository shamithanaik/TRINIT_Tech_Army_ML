


#**TRINIT_Tech_Army_ML**
Sexual Harassment Classification Model This repository contains code for a machine learning model that classifies text descriptions to detect instances of sexual harassment. The model is based on scikit-learn and uses a Random Forest classifier for multilabel classification.

#**Requirements** :
python scikit-learn pandas numpy matplotlib Installation Clone the repository to your local machine: git clone https://github.com/shamithanaik/TRINIT_Tech_Army_ML.git

#**Usage** :
Ensure that you have the necessary dataset files (train_project.csv and test_project.csv) in the project directory.

Run the Jupyter notebook multilabel.ipynb to train and evaluate the model.

Alternatively, you can directly run the Python script multilabel.py:

bash python multilabel.py

we have also added the codefile for binarylabel classification of data as binary.ipynb.

#**Dataset** :
The dataset consists of text descriptions related to sexual harassment incidents. The train_project.csv file contains the training data, while test_project.csv contains the test data.these are the same ones given to us .

#**Model Evaluation** :
The model's performance is evaluated using accuracy, precision, recall, and Hamming loss metrics. These metrics provide insights into the model's classification performance across different labels. in multilabel classification we have used 2 model - decision tree and random forest.

#**Results:**
The prediction results are saved in a CSV file named predictions_<model_name>.csv. This file contains the predicted labels for the test dataset for differnt models. a simple shap visualization has also been done.

#**deployment video:**
demo - https://drive.google.com/file/d/13xuDGcdqhz3DFwWFgSHYWjJ3Fe8OuRkk/view?usp=sharing
deployment - https://drive.google.com/file/d/13bpt4GMByThQ7uTK8e8lOEAAe56dE_KQ/view?usp=sharing



# ML-MODEL-DEPLOYMENT-USING-FLASK
this code contains pickle files and web page using flask file .
Steps to run this
1) Run model.py which stores all the machine learning models into pickle files
2) Run app.py to get the web page
