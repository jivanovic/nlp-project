# NLP Project: IMapBook Collaborative Discussions Classification

A project for the Natural Language Processing course at the Faculty of Computer and Information Science at University
of Ljubljana. 

The task is related to a project, where research on book comprehension for primary school pupils was conducted using
the IMapBook system. The pupils formed groups and could use the IMapBook system to communicate with each other and form
collaborative responses to questions related to a book they read. The main goal is to classify messages from the pupils' 
conversations taken from the IMApBook systems using NLP techniques and determine the classification performance.

## Data
Approximately 800 chat messages to be classified into predefined classes

The database also contains additional information such as message senders, group names, timestamps of messages etc.

## How to Run

The main two files to run the classifiers are main.py for the traditional approaches and main_neural.py for the BERT transformer model. The easiest way to ensure to have all of the libraries installed is to check the requirements.txt file in the project. To run the main.py file you need to download the pre-trained word2vec embedding from the following link:
https://zenodo.org/record/3237458#.YHwg75MzadY
The file you need is in the archive glove.twitter.27B.200d.txt.gz; you copy glove.twitter.27B.200d.txt from the archive into the code folder and the classifier should be ready to run.
For the deep learning method, you can adjust some settings in the main_neural.py. We propose setting the number of epochs to at least 10 as the network usually performs better if that is done. The network was trained on an NVIDIA GeForce GTX 1060 graphics card. The number of epochs can be changed on line 98 in the main_neural.py file. Make sure the weights file waights_at_finish.h5 is in the models directory.

We attempted to implement a custom network in the main_neural_custom.py file but the development was discontinued.

## The Report
A continuously updated report for the project is available in the root directory with the name "report.pdf". 

## Authors
Jan Ivanovič,  email: ji4607@student.uni-lj.si
Grega Dvoršak, email: gd4667@student.uni-lj.si
