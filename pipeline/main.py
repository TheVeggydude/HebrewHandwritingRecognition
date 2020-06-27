import argparse
import shutil
import random
import cv2
import os
import numpy as np


from statistics import mode
from imutils import paths
from tensorflow.keras.models import load_model

from character_recognizer import cfg
from preprocessor.preprocessor import get_characters_from_image

def convert_classes_to_hebrew(classes):
    hebrew = {
        '0': '\u05D0',
        '1': '\u05E2',
        '2': '\u05D1',
        '3': '\u05D3',
        '4': '\u05D2',
        '5': '\u05D4',
        '6': '\u05D7',
        '7': '\u05DB',
        '8': '\u05DA',
        '9': '\u05DC',
        '10': '\u05DE',
        '11': 'mem medial',
        '12': '\u05DF',
        '13': 'nun medial',
        '14': '\u05E4',
        '15': '\u05E3',
        '16': '\u05E7',
        '17': '\u05E8',
        '18': '\u05E1',
        '19': '\u05E9',
        '20': '\u05EA',
        '21': '\u05D8',
        '22': '\u05E5',
        '23': 'tsadi medial',
        '24': '\u05D5',
        '25': '\u05D9',
        '26': '\u05D6'
    }
    characters = []
    for character in classes:
        characters.append(hebrew.get(str(character)))

    return characters

def convert_class_to_style(result):
    styles = {
        0: 'Archaic',
        1: 'Hasmonean',
        2: 'Herodian'
    }
    return styles.get(result)

def save_result_style(results, index):
    filename = 'results/img_' + str(index) + '_style.txt'
    
    prediction = convert_class_to_style(mode(results))

    f = open(filename, 'a')
    f.write(prediction + '\n')
    f.close()

def save_results_characters(results, index):
    filename = 'results/img_' + str(index) + '_characters.txt'

    if os.path.exists(filename):
        append_write = 'a' # append if already exists
    else:
        append_write = 'w' # make a new file if not
    
    f = open(filename, append_write)
    f.write(''.join(results) + '\n')
    f.close()


def predict_chars():
    # Load character recognition model
    character_model = load_model("character_recognizer/char_model_loss")

    # Load style model
    style_model = load_model("style_classifier/model3_new.h5")

    # initialize the data and labels
    print("[INFO] loading images...")
    data = []

    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(args["dataset"])))
    random.seed(42)
    random.shuffle(imagePaths)

    image_number = -1
    # loop over the input images
    for imagePath in imagePaths:
        image_number += 1
        styles = []

        # Get characters from each image
        characters = get_characters_from_image(imagePath)

        # TODO: REVERSE ORDER OF LISTS SO IT'S HEBREW
        # Loop through every segmented line
        for line in characters:
            line_characters = []

            # Loop through every character on the line
            # Retrieved character from a line from an image
            for character in line:
                character = cv2.resize(character, (64, 64))
                line_characters.append(character)

            # (NUM_CHARS, X, Y, CHANNEL)
            data = np.array(line_characters)
            data = np.array(data, dtype="float") / 255.0
            data = np.expand_dims(data, axis=3)
            
            # Predict classes of characters from line
            y_new = np.argmax(character_model.predict(data), axis = -1)
            y_new = np.flip(y_new)

            # Convert classes to hebrew characters
            y_new = convert_classes_to_hebrew(y_new)

            # Write classes to file
            save_results_characters(y_new, image_number)

            # Predict style of image
            style = np.argmax(style_model.predict(data), axis = -1)
            style = np.ndarray.tolist(style)
            styles = styles + style
            
        # Save style of image
        save_result_style(styles, image_number)
    pass

if __name__ == "__main__":
    dir = 'results'
    if os.path.exists(dir):
        shutil.rmtree(dir)
    os.makedirs(dir)
    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset of images")

    args = vars(ap.parse_args())

    predict_chars()

    pass