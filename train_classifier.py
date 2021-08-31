# Import libraries;

import sys
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

# Load dataset:
def load_data(database_filepath):
    
    """
    Function to Load dataset from database sql (database_filepath) and split the dataframe into X and Y variables.
    
    Input: Database filepath
    Output: Returns variables X and Y along with columns names catgeory_names.
    
    """
   
    # Load database:
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql('SELECT * FROM clean_data', con = engine)
    
    # Allocate values to X and Y:
    X = df['message'].values
    Y = df[df.columns[4:]]
    category_names = df.columns[4:]
    
    return X, Y, category_names

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
    
    # Remove puncthuations and Convert to lower case:
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9]"," ",text)
    
    # get list of all urls using regex
    detected_urls = re.findall(url_regex, text)
    
    # replace each url in text string with placeholder:
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    
    # Tokenize the text:
    words = word_tokenize(text)
    
    # Remove stop words:
    tokens = [ele for ele in words if ele not in stopwords.words('english')] 
    
    # Lemmatize verbs by specifying pos:
    lemmatizer = WordNetLemmatizer()
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)

    return clean_tokens


def build_model():
    
    '''
    Function specifies the pipeline and the grid search parameters so as to build a
    classification model.
     
    Output:  cv: classification model
    
    '''
    
    # Create pipeline:
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier(n_jobs = -1)))
    ])
    
    # Choose parameters:
    parameters = {
     'clf__estimator__n_estimators': [5]
    }
    
    # Apply GridSearchCV model:
    model = GridSearchCV(pipeline, param_grid=parameters)
    
    return model


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
      Function to evaluate model and predict using test dataset and to check f1 score.
      
      input: X_test, Y_test and model.
      
      output: prints classification report .
      
    """
    # Predict model using test dataset.
    y_predict = model.predict(X_test)
    
    # Check f1 scores.
    report = classification_report(Y_test,y_predict,target_names = category_names)
    
    # Print report:
    print(report)
    
    return report


def save_model(model, model_filepath):
    pickle.dump(model, open(model_filepath, 'wb'))


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()