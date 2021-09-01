# Disaster Response Pipeline Project

## Table of Contents
1. Project Description
2. Important folders and files
3. To start
4. Graphs
5. Results
6. Acknowledgement and License

### Project Description:

* This Project is one of the Data Science Nanodegree Program of Udacity in collaboration with appen. The initial dataset contains pre-labelled tweet and messages from real-life disaster situations. The aim of the project is to build a Natural Language Processing tool that categorize messages.

* The Project is divided in the following Sections:

1. Processing Data, ETL Pipeline for extracting data from source, cleaning data and saving them in a proper database structure.
2. Machine Learning Pipeline for training a model to be able to classify text message in categories.
3. Web App for showing model results in real time.

### Important folders and files:

* process_data.py: This python code takes as input csv files(message data and message categories datasets), clean it, and then creates a SQL database
* train_classifier.py: This code trains the ML model with the SQL data base
* ETL Pipeline Preparation.ipynb: process_data.py development process
* ML Pipeline Preparation.ipynb: train_classifier.py. development process
* data: folder contains sample messages and categories datasets in csv format
* app: contains the run.py to initiate the web app.

### To start:

* Python 3.5+ (I used Python 3.7)
* Machine Learning Libraries: NumPy, SciPy, Pandas, Sciki-Learn
* Natural Language Process Libraries: NLTK
* SQLlite Database Libraries: SQLalchemy
* Web App and Data Visualization: Flask, Plotly

### Instructions:
1.  To run the following commands root directory to set up database and model.

    - run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/

### Graphs:
![image](https://user-images.githubusercontent.com/81925727/131719852-a1aa7a69-b5be-42b7-b709-04cf66c571f3.png)

![image](https://user-images.githubusercontent.com/81925727/131720238-866a5b89-fa66-412d-98ce-5258544c06c7.png)

### Results:

![image](https://user-images.githubusercontent.com/81925727/131721031-29834a7a-5ffb-4932-9597-ace94eb64129.png)


### Acknowledgement and License:
* Thanks to Udacity for providing such an excellent Data Science Nanodegree Program.
