import os

PROJECT_ROOT = os.path.abspath(os.path.dirname(__file__))

""" Sentiment labels trained on """
SENTIMENTS = ['NEG', 'NEU', 'POS']

""" Use the transformer architecture from here: 
    https://static.googleusercontent.com/media/research.google.com/en//pubs/archive/46808.pdf """
SENTENCE_ENCODER_MODEL_PATH = "https://tfhub.dev/google/universal-sentence-encoder-large/3"

""" Pretrained Keras sentiment model with USE """
SENTIMENT_MODEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'model.h5')

""" Lexicon list and emotions from: https://saifmohammad.com/WebPages/NRC-Emotion-Lexicon.htm
    Covers [anger, anticipation, disgust, fear, joy, sadness, surprise, trust] from Robert Plutchik """
EMOTIONS_MODEL_PATH = os.path.join(PROJECT_ROOT, 'data', 'emotions_lexicon.p')

BATCH_SIZE = 256