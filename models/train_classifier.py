import sys
import numpy as np
import pandas as pd
from sqlalchemy import create_engine
import re
import nltk
nltk.download('wordnet')
from nltk.stem.wordnet import WordNetLemmatizer
nltk.download('punkt')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pickle
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV

def load_data(database_filepath):
    engine = create_engine('sqlite:///{}'.format(database_filepath))
    df = pd.read_sql_table("Messages", engine)
    X = df['message']
    y = df.drop(['id', 'message', 'original', 'genre'], axis = 'columns')
    cat_names = y.columns
    return X, y, cat_names


def tokenize(text):
    # Removeing anything but words and Change all the letter to lower case
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Returning the words to 
    lemmatizer_object = WordNetLemmatizer()
    # Tokenizeing the text
    words = word_tokenize(text)
    # Deletiong stopwords
    tokens = [word for word in words if word not in stopwords.words("english")]
    
    new_tokens = []
    for token in tokens:
        new_token = lemmatizer_object.lemmatize(token).lower().strip()
        new_tokens.append(new_token)
        
    return new_tokens


def build_model():
    # Building the model
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer = tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf',MultiOutputClassifier(RandomForestClassifier()))
    ])

    parameters = {'clf__estimator__n_estimators': [14, 28],
                  'tfidf__smooth_idf': [True, False]}
    cv = GridSearchCV(pipeline, param_grid=parameters, n_jobs=-1)
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    # Evaluting the model
    print("Classification Report per Category:\n")
    y_predicted = model.predict(X_test)
    for i, c in enumerate(category_names):
        print(c)
        print(classification_report(Y_test[c], y_predicted[:,i]))

def save_model(model, model_filepath):
    # Saving the model
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
