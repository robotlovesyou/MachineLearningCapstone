# Machine Learning Capstone

The code contained in this repository was developed to be run on the Floydhub cloud machine learning service. Local Development was done using a docker container to allow an environment as close to the standard floydhub as possible to be created. Options are provided to allow the code to run as a local process as well.

## Installing and Running Under Docker

Assuming you have docker installed.

All instructions assume you have a terminal open at the project root.

1. Build the Docker image using `docker build -t mlcap .`
2. Either run the code to train and test  a 'modern' network by running the script runmod.sh `./runmod.sh`
3. Or run the code to train and test an 'original' network by running the script runorig.sh `./runorig.sh`

The various options which can be used to set hyperparameters and experimental options are explained in the section **Options** below.

## Running the code on Floydhub

All instructions assume you have a terminal open at the project root.

Assuming you have the floydhub cli installed and you are authenticated.

1. Create a new dataset on floydhub.
2. `cd dataset`
3. `floyd data init your_dataset_name`
4. `floyd data upload`
5. `cd ..`
6. Create a new project on floydhub
7. `floyd init your_project_name`
8. Edit the --data parameter used in runonfloyd.sh to match the id of your dataset
9. `runonfloyd.sh`

The various options which can be used to set hyperparameters and experimental options are explained in the section **Options** below.

## Running locally

It is recommended that the following is carried out in a virtualenv to prevent conflicts.

All instructions assume you have a terminal open at the project root.

1. `pip install -r requirements.txt`
2. `mkdir output`
3. To train and run a 'modern' network with 3 layers of 60 units per layer for 10 epochs `python main.py -t modern -e 10 -l 60 60 60 -D dataset/covtype.data -o output/`
4. To train and run an 'original' network for 10 epochs `python main.py -t original -e 10  -D dataset/covtype.data -o output/`

The various options which can be used to set hyperparameters and experimental options are explained in the section **Options** below.

## Options
The following options are used to control the type of experiment run, the input and output locations and certain hyperparameters

* -t --type. [original|modern]. required. Instruct the code to train a 'modern' network or an 'original' network
* -e --epochs. defaults to 100. Set the number of training epochs
* -r --resample [undersample|oversample]. Instruct the code to preprocess the data using either random over/under sampling. Defaults to none.
* -d --dropout. defaults to 0.5. Sets the dropout rate used per layer. Ignored when training an 'original' network
* -l --layers. Describes the number of units per hidden layer, eg `-l 60 60 60 60` to train a network with 4 hidden layers of 60 units per layer. Ignored when training an 'original' network
* -n --num_per_class. Manually set the number of samples to use when using random undersampling. Must provide one value per class (7 values) eg `-r undersample -n 1000 1100 1200 1300 1400 1500 1600`. Ignored unless -r undersample is also provided. If undersampling is used but this option is not then the number of samples of each class with be equal to the proportion of data used for training * the number of examples of the smallest minority class. Undefined if the values provided are > the number of available samples of any class
* -w --weight_data. Use class weights when training the data.
* -b --boost_per_class. Multiply the class weights calculated by sklearn by the values provided. Must provide one value per class (7 values) eg `-w -b 1.1 1.2 1.3 1.4 1.5 1.6 1.7`. Ignored unless data weighting is used.
* -s --seed. Set a fixed seed for train test splits and kernel initialization.
* -m --min_loss. Set a minimum loss value to use for early exiting from training over very large numbers of epochs.
* -D --dataset. Set the path to the data used for training and testing. Only used when running the code locally outside of docker.
* -o --outdir. Set the path to write the report, the saved model and the log of loss and accuracy figures collected during training.

### Examples
`python -t modern -e 10 -l 1080 1080 1080 -s 123456 -D dataset/covtype.data -o output/` would train and test a 'modern' network over 10 epochs with 3 layers of 1080 units each, a random seed of 123456, loading the dataset from dataset/covtype.data and writing the report to the output directory

`python -t original -e 1000 -D dataset/covtype.data -o output/` would train an 'original' network over 1000 epochs loading the dataset from dataset/covtype.data and writing the report to output/

`python -t original -e 10000 -w -b 1.25 1.5 1.5 1 1 1.25 1` would train an 'original' network over 10000 epochs using class weights with said class weights boosted by the constants provided, loading the data from the default location (suitable for running under docker or on floydhub). These are the settings used to train the benchmark network.

`python -t modern -e 1000 -l 2160 2160 2160` would train a 'modern' network with three hidden layers of 2160 units per layer over 1000 epochs. These are the settings used to train the selected network

## Files
dataset/covtype.data The dataset used for the project. Acquired from the UCI Machine Learning Repository
dataset/covtype.info The description of the dataset used for the project. Also Acquired from the UCI Machine Learning Repository
.floydignore Ignore file for floydhub detailing files which should not be uploaded. Used by the floydhub cli
.gitignore Ignore file for git detailing files which should not be included in the repository
data.py Code to load the dataset and split it into data and labels
Dockerfile docker container build file used to build the docker container used for local development
explore.py Code used to cary out some basic exploration of the dataset
floyd_requirements.txt Instructs floydhub to install extra python libraries in its environment
main.py Code to train and test models based on provided options and to create a report detailing their performance
project.pdf The capstone project report
proposal.pdf The capstone project proposal
README.md This file
requirements.txt Instructions used by pip to install dependencies required to run the project both locally and in local docker
runexplore.sh Bash script to run the explore.py file inside a docker container with appropriately mounted volumes for the dataset and output
runmod.sh Bash script to train and test a modern network inside a docker container with appropriately mounted volumes for the dataset and output. Should be edited to set the parameters for the network
runonfloyd.sh Bash script to train and test a network on floydhub. Should be edited to set the parameters for the network and for floydhub (eg dataset id)
runorign.sh Bash script to train and test an original network inside a docker container with appropriately mounted volumes for the dataset and output. Should be edited to set the parameters for the network
