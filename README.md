# toric-RL-decoder

![](src/toric_code_gif.gif)

Deep reinforcement learning decoder for the toric code

## Prerequisites 
- Python 3

## Getting started 
### Installation 
- The required libraries are matplotlib, numpy and pytorch (add all requirements!)

```bash
pip install -r requirements.txt
```

- Clone this repo:
```bash
git clone https://github.com/mats-granath/toric-RL-decoder.git
```

## How to use the simulator
There are two example scripts
- train_script.py
- prediction_script.py

The train script trains an agent to solve syndromes. All the hyperparameters related to the training are specified in the script. Moreover, an evaluation of the training run is stored in the data folder with a timestamp.

The predict script uses a trained network and predicts given a specified amount of syndromes. The trained network can be loaded from the network folder.


## Structure of this repo

File | Description
---- | -----
`├── data` | A directory that contains the evaluation data for each training or prediction run.
`├── docs` | Documentation for the different classes and functions.
`├── network` | Pretrained models for the grid sizes 5,7, and 9.
`├── plots` | All the plots generated during prediction are saved in that folder.
`├── src` | Source files for the agent and toric code.
`·   ├── RL.py` | 
`·   ├── Replay_memory.py` | Contains classes for a replay memory with uniform and proportional sampling. 
`·   ├── Sum_tree.py` | A binary tree data structure where the parent’s value is the sum of its children.
`·   └── Toric_model.py` | Contains the class toric_code and functions that are relevant to manipulate the grid.
`├── Dockerfile` | The definition for the Docker container on which the simulation executes.
`├── README.md` | About the project.
`├── entrypoint.sh` | Called inside the container to execute the simulation. Can also be used locally.
`├── requirements.txt` | The Python libraries that will be installed. Only the libraries in this official repo will be available.
`└── run.sh` | The only command you need. Builds and runs simulations in the Docker container.
