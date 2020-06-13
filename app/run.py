import json
import plotly
import pandas as pd
import numpy as np

from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize

from flask import Flask
from flask import render_template, request
from plotly.graph_objs import Bar
from sklearn.externals import joblib
from sqlalchemy import create_engine
import re

app = Flask(__name__)

def tokenize(text):
    sentence = text.lower()
    sentence = re.sub(r"[^a-zA-Z0-9]", " ", sentence)
    tokens = word_tokenize(text)
    lemmatizer = WordNetLemmatizer()
    
    lemmatize_tokens = []
    
    for token in tokens:
        clean_token = lemmatizer.lemmatize(token).lower().strip()
        lemmatize_tokens.append(clean_token)
        
        
    return lemmatize_tokens

# loading data
engine = create_engine('sqlite:///../data/DisasterResponse.db')
df = pd.read_sql_table('Processed_Message', engine)

# loading model
model = joblib.load("../models/classifier.pkl")


# index webpage displays cool visuals and receives user input text for model
@app.route('/')
@app.route('/index')
def index():

    # extract data needed for visuals
    count_genre = df.groupby('genre').count()['message']
    unique_genre = list(count_genre.index)

    # Show distribution of different category
    category = list(df.columns[4:])
    count_per_category = []
    for column_name in category:
        count_per_category.append(np.sum(df[column_name]))
    
    categories = df.iloc[:,4:]
    last10_mean_categories = categories.mean().sort_values()[0:10]
    last10_categories_names = list(last10_mean_categories.index)
    
    
    # extract data exclude related
    categories = df.iloc[:,4:]
    top10_mean_categories = categories.mean().sort_values(ascending=False)[0:10]
    top10_categories_names = list(top10_mean_categories.index)

    
    # create visuals
    graphs = [
        {
            'data': [
                Bar(
                    x=unique_genre,
                    y=count_genre
                )
            ],

            'layout': {
                'title': ' Message Genres Distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Genre"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=top10_categories_names,
                    y=top10_mean_categories
                )
            ],

            'layout': {
                'title': 'Top 10 Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=last10_categories_names,
                    y=last10_mean_categories
                )
            ],

            'layout': {
                'title': 'Least Searched Categories',
                'yaxis': {
                    'title': "Percentage"
                },
                'xaxis': {
                    'title': "Categories"
                }
            }
        },
        {
            'data': [
                Bar(
                    x=category,
                    y=count_per_category
                )
            ],

            'layout': {
                'title': 'Message Categories Distribution',
                'yaxis': {
                    'title': "Count"
                },
                'xaxis': {
                    'title': "Category"
                }
            }
        }
      
    ]

    # encode plotly graphs in JSON
    ids = ["graph-{}".format(i) for i, _ in enumerate(graphs)]
    graphJSON = json.dumps(graphs, cls=plotly.utils.PlotlyJSONEncoder)

    # render web page with plotly graphs
    return render_template('master.html', ids=ids, graphJSON=graphJSON)

# web page that handles user query and displays model results
@app.route('/go')
def go():
    # save user input in query
    query = request.args.get('query', '')

    # use model to predict classification for query
    classification_labels = model.predict([query])[0]
    classification_results = dict(zip(df.columns[4:], classification_labels))

    # This will render the go.html Please see that file.
    return render_template(
        'go.html',
        query=query,
        classification_result=classification_results
    )


def main():
    app.run(host='0.0.0.0', port=3001, debug=True)


if __name__ == '__main__':
    main()