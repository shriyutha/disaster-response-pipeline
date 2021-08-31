#!/usr/bin/env python
# coding: utf-8

# # ML Pipeline Preparation
# Follow the instructions below to help you create your ML pipeline.
# ### 1. Import libraries and load data from database.
# - Import Python libraries
# - Load dataset from database with [`read_sql_table`](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.read_sql_table.html)
# - Define feature and target variables X and Y

# In[1]:


# import libraries

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import re
import warnings
warnings.filterwarnings("ignore")
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download(['wordnet', 'averaged_perceptron_tagger'])
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem.wordnet import WordNetLemmatizer

from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.multioutput import MultiOutputClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV
import pickle


# In[2]:


# load data from database
engine = create_engine('sqlite:///DisasterResponse.db')
df = pd.read_sql("SELECT * FROM clean_data", engine)
X = df['message'].values
Y = df[df.columns[4:]]


# In[3]:


# Display the value of X:

X


# In[4]:


# Display the value of Y:

Y.head()


# In[ ]:


# Display the various columns:

df.columns


# ### 2. Write a tokenization function to process your text data

# In[5]:


url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'

def tokenize(text):
    
    """
    Outputs editted version of the input Python str object `text` 
    replacing all urls in text with str 'urlplaceholder'.
    
    Takes a Python string object and outputs list of processed words 
       of the text.
    
    INPUT:
        - text - Python str object - a raw text data
        
    OUTPUT:
        - text - Python str object - An editted version of the input data `text` 
          with all urls in text replacing with str 'urlplaceholder'.
        - tokens - Python list object - list of processed words using the input `text`.
        
    """
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    
    detected_urls = re.findall(url_regex, text)
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")

    words = word_tokenize(text)
    
    tokens = [ele for ele in words if ele not in stopwords.words('english')] 
    
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens
    


# In[6]:


# Print the function:

print(tokenize(X[4]))


# ### 3. Build a machine learning pipeline
# This machine pipeline should take in the `message` column as input and output classification results on the other 36 categories in the dataset. You may find the [MultiOutputClassifier](http://scikit-learn.org/stable/modules/generated/sklearn.multioutput.MultiOutputClassifier.html) helpful for predicting multiple target variables.

# In[8]:


pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs = -1)))
    ])


# In[9]:


pipeline.get_params()


# ### 4. Train pipeline
# - Split data into train and test sets
# - Train pipeline

# In[10]:


# Split data into train and test:

X_train, X_test, y_train, y_test = train_test_split(X, Y)


# In[11]:


# Fit model:

pipeline.fit(X_train, y_train)


# In[12]:


# Predict using test data:

y_pred = pipeline.predict(X_test)


# In[13]:


# check rows n columns:

y_pred.shape, y_test.shape, len(list(Y.columns))


# In[14]:


# Check accuracy:

labels = np.unique(y_pred)

accuracy = (y_pred == y_test).mean()

print("Labels:", labels)

print("Accuracy: \n\n", accuracy)


# from sklearn.base import BaseEstimator, TransformerMixin
# from custom_transformer import StartingVerbExtractor
# 
# feature_pipeline = Pipeline([
#         ('features', FeatureUnion([
# 
#             ('text_pipeline', Pipeline([
#                 ('vect', CountVectorizer(tokenizer=tokenize)),
#                 ('tfidf', TfidfTransformer())
#             ])),
# 
#             ('starting_verb', StartingVerbExtractor())
#         ])),
# 
#         ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs = -1)))
#     ])
# 

# ### 5. Test your model
# Report the f1 score, precision and recall for each output category of the dataset. You can do this by iterating through the columns and calling sklearn's `classification_report` on each.

# In[15]:


# Check f1 score:

print(classification_report(y_test,y_pred,target_names = df.columns[4:]))


# ### 6. Improve your model
# Use grid search to find better parameters. 

# In[16]:


pipeline.get_params().keys()


# In[17]:


# select parameters for GridSearchCV:

parameters = {
     'clf__estimator__n_estimators': [5]
    }

cv = GridSearchCV(pipeline, param_grid=parameters, cv = 3)


# cv = GridSearchCV(
#     pipeline, 
#     param_grid=parameters,
#     cv=3,
#     scoring=avg_accuracy_cv, 
#     verbose=3)
# 

# In[18]:


# Fit model:

model = cv.fit(X_train, y_train)


# In[19]:


# Predict using test data:

y_predict = model.predict(X_test)


# In[20]:


# check rows n columns:

y_predict.shape, y_test.shape, len(list(Y.columns))


# In[21]:


# Check accuracy of the model:

labels = np.unique(y_predict)

accuracy = (y_predict == y_test).mean()

print("Labels:", labels)

print("Accuracy: \n\n", accuracy)


# ### 7. Test your model
# Show the accuracy, precision, and recall of the tuned model.  
# 
# Since this project focuses on code quality, process, and  pipelines, there is no minimum performance metric needed to pass. However, make sure to fine tune your models for accuracy, precision and recall to make your project stand out - especially for your portfolio!

# In[22]:


# check f1 score:

print(classification_report(y_test,y_predict,target_names = df.columns[4:]))


# ### 8. Try improving your model further. Here are a few ideas:
# * try other machine learning algorithms
# * add other features besides the TF-IDF

# ### 9. Export your model as a pickle file

# In[23]:


filename = 'random_forest_classifier_model.pkl'
pickle.dump(pipeline, open(filename, 'wb'))


# ### 10. Use this notebook to complete `train.py`
# Use the template file attached in the Resources folder to write a script that runs the steps above to create a database and export a model based on a new dataset specified by the user.

# In[ ]:




