import argparse
import random
import cv2
import os
import numpy as np

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
        '10': '\u05DE ',
        '11': 'mem medial',
        '12': '\u05DF',
        '13': 'nun medial',
        '14': '\u05E4',
        '15': '\u05E3',
        '16': '\u05E7',
        '17': '\u05E8',
        '18': '\u05E1',
        '19': '\u05E9',
        '20': '\u05EA(taw)',
        '21': '\u05D8',
        '22': '\u05E5',
        '23': 'tsadi medial',
        '24': '\u05D5(waw)',
        '25': '\u05D9',
        '26': '\u05D6'
    }
    characters = []
    for character in np.array2string(classes):
        if character.isdigit():
            characters.append(hebrew.get(character))
    print("characters: ", characters)

    return characters

def save_results(results, index):
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
    model = load_model("character_recognizer/char_model_loss")

    # initialize the data and labels
    print("[INFO] loading images...")
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(args["dataset"])))
    random.seed(42)
    random.shuffle(imagePaths)

    image_number = -1
    # loop over the input images
    for imagePath in imagePaths:
        image_number += 1

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

            print(f"Line characters: {line_characters}")
            print(f"Length of list of characters: {len(line_characters)}")
            print(f"Character shape: {line_characters[0].shape}")
            print(f"Type of character: {type(line_characters[0])}")
            
            # (NUM_CHARS, X, Y, CHANNEL)
            data = np.array(line_characters)
            data = np.array(data, dtype="float") / 255.0
            data = np.expand_dims(data, axis=3)
            
            # Predict classes of characters from line
            y_new = np.argmax(model.predict(data), axis = -1)
            y_new = np.flip(y_new)

            # Convert classes to hebrew characters
            y_new = convert_classes_to_hebrew(y_new)

            # Print classes of characters
            print(f"Classes of characters: {y_new}")

            # Write classes to file
            save_results(y_new, image_number)
    exit()


    pass

if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset of images")

    args = vars(ap.parse_args())

    predict_chars()

    pass