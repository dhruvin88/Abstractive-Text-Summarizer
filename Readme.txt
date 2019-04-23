How to run:

1.  Download the CNN\Dataset from https://cs.nyu.edu/~kcho/DMQA/
2.  Unzip the stories folder and set the path to the folder in line 47 of preprocess_cnn.py
3.  Run model.py - to train model - parameters are on line 224-233
4.  Run test_model_output.py - set path to model file on line 17
5.  Run rename.py - to rename the output files
6.  Run eval.py - to get the rouge scores from the model