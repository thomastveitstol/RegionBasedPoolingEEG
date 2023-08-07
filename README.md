# RegionBasedPoolingEEG
This repository contains the code created for the paper “Introducing Region Based Pooling for Handling a Varied Number of EEG Channels for Deep Learning Models”, which is currently under review.

The code is organised in the following three folders:
- scripts: This is where all scripts for running experiments are located. It includes a ‘models’ folder, where all DL models are trained/validated/tested. The ‘plots’ folder contains scripts which generate the results (violin plot, heatmaps, and model selection) and visualisations of the electrode positions and channel splits. The ‘save_data’ folder contains scripts for downloading the data from Child Mind Institute, converting it to numpy arrays, as well as a script for saving the channel positions. The cleaning pipeline was implemented in MATLAB, and is available at https://github.com/hatlestad-hall/prep-childmind-eeg
- src: This is where all classes, functions etc. are implemented.
- test: Some unit tests for classes/functions implemented in src.
