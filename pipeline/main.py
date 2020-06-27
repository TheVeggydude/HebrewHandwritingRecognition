import argparse
import random
import cv2
import os
import numpy as np

from imutils import paths

from character_recognizer import cfg
from preprocessor.preprocessor import get_characters_from_image

def get_chars():
    # initialize the data and labels
    print("[INFO] loading images...")
    data = []
    labels = []

    # grab the image paths and randomly shuffle them
    imagePaths = sorted(list(paths.list_images(args["dataset"])))
    random.seed(42)
    random.shuffle(imagePaths)

    # loop over the input images
    for imagePath in imagePaths:
        
        # Get characters from each image
        characters = get_characters_from_image(imagePath)

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
        exit()


    pass

if __name__ == "__main__":

    # construct the argument parser and parse the arguments
    ap = argparse.ArgumentParser()
    ap.add_argument("-d", "--dataset", required=True,
        help="path to input dataset of images")

    args = vars(ap.parse_args())

    get_chars()

    pass