import sys
from model import SentimentAndEmotionsModel
from data.test import messages
from argparse import ArgumentParser


def run_interactively():
    model = SentimentAndEmotionsModel()
    model.load_weights()

    while True:
        try:
            sys.stdout.write('>>')
            message = input()
            result = model.predict(message)
            print(f"{result}")
        except KeyboardInterrupt:
            break


def run_tests():
    model = SentimentAndEmotionsModel()
    model.load_weights()

    for message in messages:
        message = message[0]
        result = model.predict(message)
        print(f"{message} | {result.overall_sentiment} | {result.emotions_list} | {result.emotions_associated_word}")


def run_training():
    model = SentimentAndEmotionsModel()
    model.train()


if __name__ == '__main__':

    parser = ArgumentParser(description="Run the sentiment model")
    parser.add_argument('mode', choices=['run', 'train', 'test'], default='run',
                        help='Choose the mode: run, train, or test.')

    args = parser.parse_args()

    if args.mode == 'run':
        run_interactively()

    if args.mode == 'train':
        run_training()

    if args.mode == 'test':
        run_tests()
