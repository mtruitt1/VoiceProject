# Voice Gender and Age Analyzer
 Analyzes the perceived age and gender of a voice in a given voice clip. This project makes use of Tensorflow and Keras for image classification of voice spectrograms converted from five second voice clips. Trained on 16,917 voice clips and tested on 5,638 voice clips, all extracted from Fallout 3, Fallout: New Vegas, Fallout 4, and Fallout 76. The finalized accuracy on the current model is over 98% on the training set and over 90% on the testing set. Previous training models can be found in subdirectories.
 
 Single voice clips can be imported to predict the perceived age and gender; When you run the predict function on a single voice clip using the SeparatedCategoryTester class, it will output a predictions image showing what the AI thinks your age most likely is and what gender it perceives your voice as in the three different age groups.

 This project is inspired largely by similar projects using machine learning to identify or categorize the perceived age and gender of a voice. While previous attempts at classifying the age and gender of a voice rely primarily on pitch, this program instead looks at and learns to identify certain patterns in spectrogram images which can be used to identify the resonance of a voice, in addition to the voice's pitch and variance, all of which are factors which influence the perception of a voice.
