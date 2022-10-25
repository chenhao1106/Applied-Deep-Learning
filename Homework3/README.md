## Description
In this honework, we are asked to train agents to play LunarLander and MsPacman from gym, OpenAI by using deep reinforcement learning.
The codes were modified based on the sample codes provided by teaching assistents.

## Usage
#### Install required packages
```shell
$ pip3 install -r requirements.txt
```

#### Train models
###### Train agent to play LunarLander by policy gradient.
```shell
$ python3 main.py --train_pg
```
###### Train agent to play MsPacman by Deep Q-Learning.
```shell
$ python3 main.py --train_dqn
```

#### Play games
###### LuanrLander
```shell
$ python3 main.py --test_pg [--do_render]
```
###### MsPacman
```shell
$ python3 main.py --test_dqn [--do_render]
```
