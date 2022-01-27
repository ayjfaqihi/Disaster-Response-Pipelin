import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    # Loding messages_filepath file
    messages = pd.read_csv(messages_filepath)
    # Loding categories_filepath file
    categories = pd.read_csv(categories_filepath)
    # Merge the two files
    df = pd.merge(categories,messages,on='id',how='inner')
    return df


def clean_data(df):
    # Split categories into separate category columns.
    categories = df['categories'].str.split(pat=';', expand=True)
    row = categories.iloc[0]
    category_colnames = [x[:-2] for x in row]
    categories.columns = category_colnames
    for column in categories:
        categories[column] = categories[column].str[-1]
       
    df.drop('categories', axis='columns' , inplace=True)
    # concatenating the original dataframe with the new `categories` dataframe
    df = pd.concat([df,categories], axis=1)
    
    df = df[df['related'] != 2]
    # droping duplicates
    df.drop_duplicates(inplace = True)
    
    return df

def save_data(df, database_filename):
    # Saving the data in database
    engine = create_engine('sqlite:///{}'.format(database_filename))
    df.to_sql('Messages', engine, index=False)  


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