# Speech Vision
Speech Vision (SV) is  a Dysarthric Speech Recognition System  that adopts a novel approach towards dysarthric ASR in which speech features are extracted visually, then SV learns to  see the shape of the words pronounced by dysarthric individuals.

There are two folders:

1.	DysarthricCNNRezaTransferLearning folder that includes models trained with normal speech,
2.	DysarthricCNNRezaTransferLearningSD includes transfer learning and Dysarthric models. This is where you need to re-train the models with additional syntactic generated data. Look at “Train_Test.ipynb” notebook to start.

Please note if you get errors when running or testing the pre-trained models, the trained models are not compatible with your CPU/GPU and both control and dysarthric models need to be retrained from scratch. 

To setup the environment, create a python 3.6 environment and install jupyter notebook. Then run “Install packages.ipynb” notebook from the environment - it installs the required packages for you.

The dysarhtic speech samples are from UA Speech (http://www.isle.illinois.edu/sst/data/UASpeech/).
