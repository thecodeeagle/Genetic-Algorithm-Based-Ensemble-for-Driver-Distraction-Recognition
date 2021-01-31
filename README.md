# Genetic Algorithm(GA) Based Ensemble : Distracted Driver Detection
The current repository, based on my project on driver distraction detection, proposes a solution for the existing problem by constructing an "ensemble" or "comination" of deep learning networks, consisting of :
* AlexNet
* Vanilla CNN
* Hierarchical Model of DenseNet-201( Adopted from YogaPose Variations)
* EfficientNet B0
* InceptionV3 Coupled with BiDirectional LSTM's
* Original VGG16 as feature extractor
The ideology lying behind is the fact that coupling several weak classifiers together can achieve a higher performance and serve as a strong classifier, as opposed to any one of them.
# Use of Genetic Algorithm
A genetic algorithm is a search heuristic that is inspired by Charles Darwin’s theory of natural evolution. This algorithm reflects the process of natural selection where the fittest individuals are selected for reproduction in order to produce offspring of the next generation. This notion can be applied for a search problem. My main motive behind using this particular algorithm was for calculating the proper weights given to each architetcure while constructing the ensemble.

# Dataset
We made use of the AUC Distracted Driver Datasetm obtained using license agreement available at : https://heshameraqi.github.io/distraction_detection.

# Technologies Used
* Keras
* Scikit-Learn
* Python

# Getting Started
To try the code for yourself, follow the steps as below:
* There are 7 .ipynb files or Colab Notebooks, which can be run individually for a particular task.
* GAEnsemble.ipynb is the main file containing the code for running the final ensemble. You need to pretrain the six branches on dataset to be tested before actually running the ensemble for optimal results.

