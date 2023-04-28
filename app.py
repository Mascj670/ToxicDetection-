from flask import Flask, request, jsonify, render_template
import joblib
# import numpy as np
import pandas as pd
# import matplotlib.pyplot as plt
import logging
# from my_measures import BinaryClassificationPerformance
from sklearn.feature_extraction.text import HashingVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from scipy.sparse import csr_matrix, hstack
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from flask_cors import CORS, cross_origin
from flask import send_from_directory

# from sklearn.preprocessing import MaxAbsScaler
fitted_transformations = []


# function that takes raw data and completes all preprocessing required before model fits
def process_raw_data(fn, my_random_seed, test=False, toxic_data_=None):
    # read and summarize data
    if toxic_data_ is not None:
        toxic_data = toxic_data_
    else:
        toxic_data = pd.read_csv(fn)
    if (not test):
        # add an indicator for any toxic, severe toxic, obscene, threat, insult, or indentity hate
        toxic_data['any_toxic'] = (
                toxic_data['toxic'] + toxic_data['severe_toxic'] + toxic_data['obscene'] + toxic_data['threat'] +
                toxic_data['insult'] + toxic_data['identity_hate'] > 0)
    print("toxic_data is:", type(toxic_data))
    print("toxic_data has", toxic_data.shape[0], "rows and", toxic_data.shape[1], "columns", "\n")
    print("the data types for each of the columns in toxic_data:")
    print(toxic_data.dtypes, "\n")
    print("the first 10 rows in toxic_data:")
    print(toxic_data.head(5))
    if (not test):
        print("The rate of 'toxic' Wikipedia comments in the dataset: ")
        print(toxic_data['any_toxic'].mean())

    # vectorize Bag of Words from review text; as sparse matrix
    if (not test):  # fit_transform()
        hv = HashingVectorizer(n_features=2 ** 17, alternate_sign=False)
        X_hv = hv.fit_transform(toxic_data.comment_text)
        fitted_transformations.append(hv)
        print("Shape of HashingVectorizer X:")
        print(X_hv.shape)
    else:  # transform()
        X_hv = fitted_transformations[0].transform(toxic_data.comment_text)
        print("Shape of HashingVectorizer X:")
        print(X_hv.shape)

    # http://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfTransformer.html
    if (not test):
        transformer = TfidfTransformer()
        X_tfidf = transformer.fit_transform(X_hv)
        fitted_transformations.append(transformer)
    else:
        X_tfidf = fitted_transformations[1].transform(X_hv)

    # create additional quantitative features

    toxic_data['uppercase_count'] = toxic_data['comment_text'].apply(lambda x: sum(1 for c in x.split() if c.isupper()))
    toxic_data['exclamation_count'] = toxic_data['comment_text'].str.count("!")
    toxic_data['question_count'] = toxic_data['comment_text'].str.count("\?")
    print(toxic_data['uppercase_count'].mean())
    print("* * * * * * * * * * * * * ")
    print(toxic_data['exclamation_count'].mean())
    print("* * * * * * * * * * * * * ")
    print(toxic_data['question_count'].mean())
    print("* * * * * * * * * * * * * ")

    # Identify comments containing threatening language
    toxic_data['contains_threat'] = toxic_data['comment_text'].str.contains('threat', case=False, regex=False) | \
                                    toxic_data['comment_text'].str.contains('kill', case=False, regex=False) | \
                                    toxic_data['comment_text'].str.contains('murder', case=False, regex=False) | \
                                    toxic_data['comment_text'].str.contains('shoot', case=False, regex=False) | \
                                    toxic_data['comment_text'].str.contains('slaughter', case=False, regex=False) | \
                                    toxic_data['comment_text'].str.contains('KYS', case=False, regex=False) | \
                                    toxic_data['comment_text'].str.contains('rape', case=False, regex=False) | \
                                    toxic_data['comment_text'].str.contains('punish', case=False, regex=False) | \
                                    toxic_data['comment_text'].str.contains('terrorize', case=False, regex=False) | \
                                    toxic_data['comment_text'].str.contains('Extort', case=False, regex=False) | \
                                    toxic_data['comment_text'].str.contains('kidnap', case=False, regex=False) | \
                                    toxic_data['comment_text'].str.contains('attack', case=False, regex=False)
    toxic_data['contains_threat'] = toxic_data['contains_threat'].astype(int)
    print("The rate of 'threat' in the dataset: ")
    print(toxic_data['contains_threat'].mean())

    # Identify comments containing rudeness
    toxic_data['contains_rudeness'] = toxic_data['comment_text'].str.contains('dumb', case=False, regex=False) | \
                                      toxic_data['comment_text'].str.contains('idiot', case=False, regex=False) | \
                                      toxic_data['comment_text'].str.contains('stupid', case=False, regex=False) | \
                                      toxic_data['comment_text'].str.contains('moron', case=False, regex=False) | \
                                      toxic_data['comment_text'].str.contains('pathetic', case=False, regex=False) | \
                                      toxic_data['comment_text'].str.contains('loser', case=False, regex=False) | \
                                      toxic_data['comment_text'].str.contains('arrogant', case=False, regex=False) | \
                                      toxic_data['comment_text'].str.contains('shut up', case=False, regex=False)
    toxic_data['contains_rudeness'] = toxic_data['contains_rudeness'].astype(int)
    print("The rate of 'rudeness' in the dataset: ")
    print(toxic_data['contains_rudeness'].mean())

    # Identify comments containing hate speech
    toxic_data['contains_hate'] = toxic_data['comment_text'].str.contains('nazi', case=False, regex=False) | \
                                  toxic_data['comment_text'].str.contains('nigger', case=False, regex=False) | \
                                  toxic_data['comment_text'].str.contains('nigga', case=False, regex=False) | \
                                  toxic_data['comment_text'].str.contains('spic', case=False, regex=False) | \
                                  toxic_data['comment_text'].str.contains('snowflake', case=False, regex=False) | \
                                  toxic_data['comment_text'].str.contains('beaner', case=False, regex=False) | \
                                  toxic_data['comment_text'].str.contains('tranny', case=False, regex=False) | \
                                  toxic_data['comment_text'].str.contains('faggot', case=False, regex=False) | \
                                  toxic_data['comment_text'].str.contains('retard', case=False, regex=False) | \
                                  toxic_data['comment_text'].str.contains('terrorist', case=False, regex=False) | \
                                  toxic_data['comment_text'].str.contains('fag', case=False, regex=False)
    toxic_data['contains_hate'] = toxic_data['contains_hate'].astype(int)
    print("The rate of 'hate speech' in the dataset: ")
    print(toxic_data['contains_hate'].mean())

    # Identify comments containing profanity
    toxic_data['contains_profanity'] = toxic_data['comment_text'].str.contains('fuck', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('shit', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('asshole', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('ass', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('damn', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('bitch', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('pussy', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('whore', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('slut', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('cunt', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('hoe', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('dick', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('cock', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('twat', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('porn', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('prick', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('cunt', case=False, regex=False) | \
                                       toxic_data['comment_text'].str.contains('hell', case=False, regex=False)
    toxic_data['contains_profanity'] = toxic_data['contains_profanity'].astype(int)

    print("The rate of 'profanity' in the dataset: ")
    print(toxic_data['contains_profanity'].mean())

    # features from Amazon.csv to add to feature set
    toxic_data['word_count'] = toxic_data['comment_text'].str.split(' ').str.len()
    toxic_data['punc_count'] = toxic_data['comment_text'].str.count("\.")

    X_quant_features = toxic_data[
        ["word_count", "punc_count", "contains_threat", "contains_rudeness", "contains_hate", "contains_profanity"]]
    print("Look at a few rows of the new quantitative features: ")
    print(X_quant_features.head(10))

    # Combine all quantitative features into a single sparse matrix

    X_quant_features_csr = csr_matrix(X_quant_features)
    X_threat = csr_matrix(toxic_data['contains_threat']).transpose()
    X_rude = csr_matrix(toxic_data['contains_rudeness']).transpose()
    X_profanity = csr_matrix(toxic_data['contains_profanity']).transpose()
    X_hate = csr_matrix(toxic_data['contains_hate']).transpose()
    X_combined = hstack([X_tfidf, X_quant_features_csr, X_rude, X_threat, X_profanity, X_hate])
    X_matrix = csr_matrix(X_combined)  # convert to sparse matrix
    print("Size of combined bag of words and new quantitative variables matrix:")
    print(X_matrix.shape)

    # Create `X`, scaled matrix of features
    # feature scaling
    if (not test):
        sc = StandardScaler(with_mean=False)
        X = sc.fit_transform(X_matrix)
        fitted_transformations.append(sc)
        print(X.shape)
        y = toxic_data['any_toxic']
    else:
        X = fitted_transformations[2].transform(X_matrix)
        print(X.shape)

    # Create Training and Test Sets
    # enter an integer for the random_state parameter; any integer will work
    if (test):
        X_submission_test = X
        print("Shape of X_test for submission:")
        print(X_submission_test.shape)
        print('SUCCESS!')
        return (toxic_data, X_submission_test)
    else:
        X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = train_test_split(X, y, toxic_data, test_size=0.2,
                                                                                     random_state=my_random_seed)
        print("Shape of X_train and X_test:")
        print(X_train.shape)
        print(X_test.shape)
        print("Shape of y_train and y_test:")
        print(y_train.shape)
        print(y_test.shape)
        print("Shape of X_raw_train and X_raw_test:")
        print(X_raw_train.shape)
        print(X_raw_test.shape)
        print('SUCCESS!')
        return (X_train, X_test, y_train, y_test, X_raw_train, X_raw_test)


# Load the trained model
model = joblib.load('toxic_model.pkl')
train_path = './toxiccomments_train.csv'
X_train, X_test, y_train, y_test, X_raw_train, X_raw_test = process_raw_data(fn=train_path, my_random_seed=99)

# Define the Flask app
app = Flask(__name__)
CORS(app)

# Define the logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
handler = logging.FileHandler('app.log')
handler.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)


# Define the '/predict' endpoint to handle POST requests
@app.route('/predict/', methods=['POST'])
@cross_origin()
def predict():
    # Get the user input from the request data
    input_data = request.get_json()
    input_text = input_data['text']

    input_df = pd.DataFrame([['fqasafsfsafwqfasf',
                              input_text]],
                            columns=['id', 'comment_text'])
    print(input_df)

    raw_data, x_test_submission = process_raw_data(fn='', my_random_seed=99, test=True, toxic_data_=input_df)

    # Make a prediction using the loaded model
    prediction = model.predict(x_test_submission)

    # Log the prediction result
    logger.info(f"Predicted: {prediction[0]}, Input Text: {input_text}")
    print(f"Predicted: {prediction[0]}, Input Text: {input_text}")

    # Return the prediction result as a JSON object
    result = {'prediction': int(prediction[0])}
    return jsonify(result)


# @app.route('/confidence/', methods=['POST'])
# @cross_origin()
# def confidence():
#     # Get the user input from the request data
#     input_data = request.get_json()
#     input_text = input_data['text']
#
#     input_df = pd.DataFrame([['fqasafsfsafwqfasf',
#                               input_text]],
#                             columns=['id', 'comment_text'])
#     print(input_df)
#
#     raw_data, x_test_submission = process_raw_data(fn='', my_random_seed=99, test=True, toxic_data_=input_df)
#
#     # Make a prediction using the loaded model
#     prediction = model.predict(x_test_submission)
#
#     # Calculate the confidence score (probability)
#     confidence_score = float(prediction[0])
#
#     rounded_confidence_score = round(confidence_score, 3)
#
#     logger.info(f"Predicted: {rounded_confidence_score}, Input Text: {input_text}")
#     print(f"Predicted: {rounded_confidence_score}, Input Text: {input_text}")
#
#     # Return the rounded confidence score as a JSON object
#     result = {'confidence_score': rounded_confidence_score}
#     return jsonify(result)

# @app.route('/predict/', methods=['POST'])
# @cross_origin()
# def predict():
#     # Get the user input from the request data
#     input_data = request.get_json()
#     input_text = input_data['text']
#
#     input_df = pd.DataFrame([['fqasafsfsafwqfasf',
#                               input_text]],
#                             columns=['id', 'comment_text'])
#
#     raw_data, x_test_submission = process_raw_data(fn='', my_random_seed=99, test=True, toxic_data_=input_df)
#
#     # Make a prediction using the loaded model
#     prediction = model.predict(x_test_submission)
#
#     # Get the probability of each class
#     probabilities = model.predict_proba(x_test_submission)
#
#     # Extract the confidence score for the positive class
#     confidence_score = probabilities[0][1]
#     rounded_confidence_score = round(confidence_score, 3)
#
#     # Log the prediction result
#     logger.info(f"Predicted: {prediction[0]}, Confidence Score: {rounded_confidence_score}, Input Text: {input_text}")
#
#     # Return the prediction result and confidence score as a JSON object
#     result = {'prediction': int(prediction[0]), 'confidence_score': rounded_confidence_score}
#     return jsonify(result)

# Define the endpoint to serve the HTML file
@app.route('/')
def index():
    return render_template('index.html')


# Define the endpoint to serve the HTML file
@app.route('/toxic')
def toxic():
    return render_template('toxic.html')


# Define the endpoint to serve static assets (CSS and JS files)
@app.route('/static/<path:path>')
def send_static(path):
    return send_from_directory('static', path)


# Run the Flask app
if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8000, threaded=True,
            use_reloader=True, use_debugger=True,
            passthrough_errors=True)
