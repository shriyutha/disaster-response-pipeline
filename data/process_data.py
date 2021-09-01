# Import libraries:

import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine

import warnings
warnings.filterwarnings('ignore')  


def load_data(messages_filepath, categories_filepath):
    
    '''
    Function to load messages and categories data set from csv files and merge them  
    into a single dataframe named as df variable and returns the merged dataset.
    
    Input: messages_filepath, categories_filepath
    
    Output: - df - merged dataframe containing messages and categories dataset.
    
    '''
    
    # Load the datasets - 
    # Load message_filepath and categories_filepath to messages and categories:
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    
    # Merge both data sets using common 'id':
    df = messages.merge(categories, on = 'id')
    
    # Display the first five values of the dataset:
    df.head()
    
    # Return data:
    return df



def clean_data(df):
    
    """
     Split the values in the categories column ; character so that each value becomes a separate column. 
     Use the first row of categories dataframe to create column names for the categories data.
     Rename columns of categories with the new column names.
     
    """
    
    # Split the values in the categories column on ';' :
    categories = df['categories'].str.split(pat = ';', expand = True)
    
    # categories column names were not readable because they are splitted.
    # Select first row of the cstegories column:
    row = categories[:1]
    
    # Apply lambda to extract only names:
    extracted_list = lambda ele: ele[0][:-2]
    category_colnames = list(row.apply(extracted_list))
    
    # Rename the column to categories:
    categories.columns = category_colnames
    
    for column in categories:
        
        # Apply lambda to set each value to be last character of the string:
        categories[column] = categories[column].apply(lambda ele: ele[-1])
        # Convert to integer:
        categories[column] = categories[column].astype(int)
    
    # Change all values not equal to 0 and 1 to 1:
    for ele in categories.columns:
        categories.loc[(categories[ele] != 0) & (categories[ele] != 1), ele] = 1
    
    # Drop categories column:
    df.drop('categories', axis = 1, inplace = True)
    
    # Concat both df and categories column together:
    df = pd.concat([df, categories], axis = 1)
    
    # Drop dulicated values:
    df.drop_duplicates(inplace = True)
    
    return df


def save_data(df, database_filename):
    
    '''
    Function to save the cleaned dataframe into a sql database with file name 'clean_data'
    
    Input:  - df, database_filename
    Output: - SQL Database 
    
    '''
    engine = create_engine('sqlite:///'+ database_filename)
    df.to_sql('clean_data', engine, index=False, if_exists='replace') 


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')

if __name__ == '__main__':
    main()