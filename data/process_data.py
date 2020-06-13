import sys
import pandas as pd
from sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):
    '''
    loading the data and returning dataframe
    '''
    try:
        messages = pd.read_csv(messages_filepath)
        categories = pd.read_csv(categories_filepath, sep = ',')
        df = pd.merge(messages, categories, on = 'id')
        return df
    except:
        print("file path not found")
        
def clean_data(df):
    '''
    Cleaning the data and reshaping the columns
    '''
    categories = df['categories'].str.split(';',expand = True)
    row = categories.loc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].astype(str).str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
    
    df.drop('categories',axis = 1, inplace = True)
    df = pd.concat([df, categories], axis = 1)
    if df.duplicated().sum() != 0:
        df.drop_duplicates(inplace = True)
        
    return df

def save_data(df, database_filename):
    engine = create_engine('sqlite:///'+database_filename)
    df.to_sql('Processed_Message', engine, index=False)


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
              'data/disaster_messages.csv data/disaster_categories.csv '\
              'data/DisasterResponse.db')


if __name__ == '__main__':
    main()
