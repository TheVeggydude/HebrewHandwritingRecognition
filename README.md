# Hebrew Character Recognition & Style Classification
This repository will contain code to preprocess, recognize characters and classify styles Hebrew texts like the Dead Sea
Scrolls. The code is free to use, however, the data will not be provided as it is part of a private collection.

## Run instructions
1. `sudo -H pip3 install --upgrade pip # latest pip3`
2. `git clone https://github.com/TheVeggydude/HebrewHandwritingRecognition.git`
3. `cd HebrewHandwritingRecognition/`
4. `pip3 install -r pipeline/requirements.txt --no-index --find-links file:///tmp/packages`

Because of the NDA no images are on the repository. Move all `jpg` testing images of the form `img_0, img1,..., imgN` inside the `image-data` folder.

5. `./test.sh`

After running `./test.sh` the program will read every image in the `image-data` folder one by one and segment it into characters. The characters are fed into the character recognition model found in the`character_recognizer/char_model_loss` folder. Each character prediction of an image is written in a separate text file in the `results` folder. The segmented characters are also fed into the style recognition model found at `style_classifier/model3_new.h5`. Each character is used to classify the style of the image, so each character generates a prediction. The predictions are aggregated into a list and the `mode` of the list is the official style of the image.

## Modules
### Preprocessor
The preprocessor takes binarized images of the source data and turns them into individual characters. This is done 
through first segmenting the document into lines of text.

#### Line segmentation
First, compute pixel projection along the y-axis of the document. Next, look for points of interest to start segmentation
lines at.
