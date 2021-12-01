# Effective weight decoder

Contains source code for https://arxiv.org/abs/xxxx

High accuracy decoding scheme based on minimal information about shortest chain weights.

Karl Hammar, Alexei Orekhov, Patrik Wallin Hybelius, Anna Katariina Wisakanto, Basudha Srivastava, Anton Frisk Kockum, Mats Granath

extra code thanks to:
also the previous authors for this code and also candidate group

## Prerequisites 
- Python 3
- do we work on windows????

## Getting started 
### Installation 
- Clone the repository:
```bash
git clone git@github.com:QEC-project-2020/EWD-QEC.git
```
- Install the requirements

```bash
pip install -r requirements.txt
```
## How to use the decoder

To run a simple test of the decoder, the following bash script is provided:
- run_test.sh

This script will automatically set up the relevant environment variables and run a test of the decoder.

The test script will call upon:
- generate_data.py

which is the main data generation script. This script will in turn run the desired decoder for the desired number of steps on the desired code. This is where syndroms are generated and data is saved.

The high level decoding code is found in ***decoders.py***, and the more fine-grained details of the MCMC simulations and code definitions can be found in the ***src*** directory.

The data is saved in *pandas* DataFrames, where the first row contains a copy of the parameter set used for the simulation, and the following rows contain the generated error chain and decoder predictions. An example on how to read the file can be seen in:
- plot_example.py

### Use of MWPM
The MWPM decoder (or the eMWPM flavour) can be used in itself or as a tool to find an initial chain for other algorithm to use. The MWPM decoder is based on *blossom5* algorigithm which is available at https://pub.ist.ac.at/~vnk/software.html (for research purposes only). Download and extract the tarball in the ***src*** directory.
## Structure of this repo

File | Description
----- | -----
`├── data` | A directory that contains the evaluation data for each training or prediction run.
`├── docs` | Documentation for the different classes and functions.
`├── network` | Pretrained models for the grid sizes 5,7, and 9.
`├── plots` | All the plots generated during prediction are saved in that folder.
`├── src` | Source files for the agent and toric code.
`·   ├── RL.py` | 
`·   ├── Replay_memory.py` | Contains classes for a replay memory with uniform and proportional sampling. 
`·   ├── Sum_tree.py` | A binary tree data structure where the parent’s value is the sum of its children.
`·   └── Toric_model.py` | Contains the class toric_code and functions that are relevant to manipulate the grid.
`├── NN.py` | Contains different network architectures
`├── README.md` | About the project.
`├── ResNet.py` | Contains different ResNet architectures
`├── requirements.txt` | The Python libraries that will be installed. Only the libraries in this official repo will be available.
`├── train_script` | A script training an agent to solve the toric code.
`└── prediction_script` | The trained agent solves syndromes.