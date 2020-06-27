# Hebrew Character Recognition & Style Classification
This repository will contain code to preprocess, recognize characters and classify styles Hebrew texts like the Dead Sea
Scrolls. The code is free to use, however, the data will not be provided as it is part of a private collection.

## Run instructions
1. Ensure that the latest pip3 version is installed.

## Modules
### Preprocessor
The preprocessor takes binarized images of the source data and turns them into individual characters. This is done 
through first segmenting the document into lines of text.

#### Line segmentation
First, compute pixel projection along the y-axis of the document. Next, look for points of interest to start segmentation
lines at.
