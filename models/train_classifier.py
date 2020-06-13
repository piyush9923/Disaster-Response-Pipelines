# import libraries
import sys
import nltk
nltk.download('punkt')
nltk.download('wordnet')

import pandas as pd
from sqlalchemy import create_engine
from sklearn.externals import joblib
import re

from nltk import pos_tag
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize

from sklearn.feature_extraction.text import TfidfTransformer, CountVectorizer
from sklearn.pipeline import Pipeline
from sklearn.grid_search import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report


# defining functions
def load_data(database_filepath):
    engine = create_engine('sqlite:///'+database_filepath)
    df = pd.read_sql('Processed_Message',engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    return X,y,category_names
 
def tokenize(text):
    '''
    returns the lemmatized labels.
    '''
    # normalizing, tokenizing, lemmatizing the sentence
    sentence = text.lower()
    sentence = re.sub(r"[^a-zA-Z0-9]", " ", sentence)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    lemmatize_tokens = []
    
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        lemmatize_tokens.append(clean_token)
        
        
    return lemmatize_tokens

def build_model():
    '''
    Building model with grid search
    '''
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
    ])
    
    parameters = {
        'vect__stop_words': ['english',None],
        'tfidf__use_idf' :[True, False]
    }

    cv = GridSearchCV(pipeline, parameters, n_jobs=-1)
    return cv

def evaluate_model(model, X_test, y_test, category_names):
    y_pred = model.predict(X_test)
    for i, col in enumerate(y_test.columns): 
        print('\t\t',col,'\t\t')
        print(classification_report(y_test.iloc[:,i], y_pred[:,i]))

def save_model(model, model_filepath):
    joblib.dump(model.best_estimator_, model_filepath)
    

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, y, category_names = load_data(database_filepath)
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
       
        print('This might take some time')

        print('Training model...')
        model.fit(X_train, y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, y_test, category_names)

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