import os
import pandas as pd
import regex as re
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
import keras.layers as layers
import pickle
from keras.models import Model
from keras import backend as K
#from cai_logging import get_logger
from settings import SENTIMENTS, EMOTIONS_MODEL_PATH, SENTENCE_ENCODER_MODEL_PATH, SENTIMENT_MODEL_PATH, PROJECT_ROOT, \
    BATCH_SIZE

#logger = get_logger('Sentiment Model')
EMOTIONS_DICT = pickle.load(open(EMOTIONS_MODEL_PATH, 'rb'))

EMBED = hub.Module(SENTENCE_ENCODER_MODEL_PATH)
EMBED_SIZE = EMBED.get_output_info_dict()['default'].get_shape()[1].value


def get_dataframe(filename):
    """Might make this a static method of the class,
    but its getting big and unwieldy already"""

    with open(filename, 'r') as f:
        data = []
        for line in f.readlines():
            label, sep, text = line.partition(' ')
            if sep == '':
                continue
            text = re.sub('[^A-Za-z0-9 ,\?\'\"-._\+\!/\`@=;:]+', '', text)
            if label not in SENTIMENTS:
                continue
            data.append([label, text])

    df = pd.DataFrame(data, columns=['label', 'text'])
    df.label = df.label.astype('category')
    return df


class SentimentAndEmotionsModel:

    def __init__(self):
        self._init_tf_session()
        self.model = self._init_model()

    @staticmethod
    def _init_tf_session():
       # logger.info('Initialising tensorflow session')
        session = tf.Session()
        K.set_session(session)
        session.run(tf.global_variables_initializer())
        session.run(tf.tables_initializer())

    def _init_model(self):
        #logger.info('Initialising Keras model for 1-layer feed-forward Neural Net')
        category_counts = len(SENTIMENTS)
        input_text = layers.Input(shape=(1,), dtype=tf.string)
        embedding = layers.Lambda(self._get_universal_embedding, output_shape=(EMBED_SIZE,))(input_text)
        dense = layers.Dense(BATCH_SIZE, activation='relu')(embedding)
        pred = layers.Dense(category_counts, activation='softmax')(dense)
        model = Model(inputs=[input_text], outputs=pred)
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        return model

    @staticmethod
    def _get_universal_embedding(x):
        """Wrap embedding in a Keras lambda layer and cast explicitly as a string."""
        return EMBED(tf.squeeze(tf.cast(x, tf.string)), signature='default', as_dict=True)['default']

    @staticmethod
    def _get_emotion(model_sentiment, words):
        """ If overall sentiment is POS match with POS words in utterance and
            return associated emotions. Only return emotions for first word.
            In the future we should perform some kind of average. """
        words = words.split()
        for w in words:
            try:
                emotion_dict = EMOTIONS_DICT[w.lower()]
            except KeyError:
                continue

            word_sentiment = emotion_dict.get('sentiment')
            word_emotion = emotion_dict.get('emotion')

            if EMOTIONS_DICT.get(w) and word_sentiment == model_sentiment:
                return w, word_emotion
        return '', 'no emotions mapped'

    def _get_sentiment(self, message):
        """ Make predictions. Need to take in a list of minimum length 2 since we trained in
        batch mode (cant find a cleaner workaround). """
        message = [message, message] 
        array_messages = np.array(message, dtype=object)[:, np.newaxis]
        predicts = self.model.predict(array_messages, batch_size=2)
        predict_logits = predicts.argmax(axis=1)
        predict_labels = [SENTIMENTS[logit] for logit in predict_logits]
        return predict_labels[0]

    def load_weights(self):
        return self.model.load_weights(SENTIMENT_MODEL_PATH)

    def train(self):
        _path = os.path.join(PROJECT_ROOT, 'data', 'train.txt')
        df_train = get_dataframe(_path)
        train_text = df_train['text'].to_numpy(dtype=object)[:, np.newaxis]
        train_label = np.asarray(pd.get_dummies(df_train.label), dtype=np.int8)
        self.model.fit(train_text,
                       train_label,
                       epochs=10,
                       batch_size=BATCH_SIZE,
                       class_weight='auto')
        self.model.reset_states()
        self.model.save_weights('model.h5')

    def predict(self, message):
        sentiment = self._get_sentiment(message)
        word, emotion = self._get_emotion(sentiment, message)
        return SentimentAndEmotionsResult(sentiment, emotion, word)


class SentimentAndEmotionsResult:
    """ Wrap sentiment, with detected emotions,
        with word which triggered emotion into a class """

    def __init__(self, overall_sentiment, emotions_list, emotions_associated_word):
        self.overall_sentiment = overall_sentiment
        self.emotions_list = emotions_list
        self.emotions_associated_word = emotions_associated_word

    def __repr__(self):
        return f"{self.overall_sentiment}; {self.emotions_list}; {self.emotions_associated_word}"
