# Self-Modeling Reinforcement Learning Machine
For UCL MSc Robotics and Compuatation Dissertation


## 0. Abstract
There have been considerable researches on the self-awareness and consciousness of the human race system in various fields. Recently, attempts have been made to simulate the establishment of self-model at the level of robot applications, but most of the existing research remains under supervision or fully observation. Thus, we believe that there is still a gap with the true self-modeling machines. This project focuses on the exploration of building a self-aware intelligent agent from pixel-level observations and with all unsupervised approaches. In addition to existing studies on environment modelling and opponent modelling, this work tries to complete the model-based RL framework by considering the learning of self models. After identifying ”itself” through their own reasonable unique learned state, our proposed agent is expected to finish the touch and pick task when applying to the three dimension 7-DoF robotic-arm platform. This work will also consider the link between related fields like philosophy, psychology, neuroscience and engineering implementation of self-awareness.


## 1. Installation

####  1.1 Create Environment
``` Bash
# create conda environment
conda create -n smrl python==3.6
conda activate smrl
pip install -r requirements.txt
conda install pytorch torchvision torchaudio cudatoolkit=11.1 -c pytorch -c nvidia
```

####  1.2 Install Pybullet3.0
Install pybullet accoring to [bullet3](https://github.com/bulletphysics/bullet3) and pybullet offical [website](https://pybullet.org/wordpress/).
``` Bash
# Install pybullet3.0
pip3 install pybullet --upgrade --user
python3 -m pybullet_envs.examples.enjoy_TF_AntBulletEnv_v0_2017may
python3 -m pybullet_envs.examples.enjoy_TF_HumanoidFlagrunHarderBulletEnv_v1_2017jul
python3 -m pybullet_envs.deep_mimic.testrl --arg_file run_humanoid3d_backflip_args.txt
```


### Inner State Trainining Test
Check whether it is running normally, this will only run a few rounds for running verification:
```
 python train_predicter_collect_by_ddpg.py --num_samples 10 --num_iters 10 --batch_size 2 --log_interval 5 --replay_buffer_capacity 100  --num_envs 1
```

## 2. Train

### 2.1. Integrated version:
Using the random exploration and active inference for policy trainning:
```
 python main.py 
```
Part of the hyperparameters represented by the learning state， please check the function main.parse_args().
###  2.2. Inner State Trainning:

#### 2.2.1 Random Sampling

```
 python train_predicter_collect_by_ddpg.py --mode random --num_samples 100000 --num_iters 1000 --batch_size 256 --log_interval 5 --replay_buffer_capacity 1000  --num_envs 1
```

#### 2.2.2 DDPG Expert Trajectory Sampling

Firstly, using DDPG to learning a trajectory:
```
 python train_predicter_collect_by_ddpg.py --mode train
 ```
 
Then, train inner state using DDPG exper trajectory:
```
 python train_predicter_collect_by_ddpg.py --mode load --num_samples 100000 --num_iters 1000 --batch_size 256 --log_interval 5 --replay_buffer_capacity 1000  --num_envs 1
```


### 2.3. Policy Learning:
#### 2.3.1 Active Inference
```
python train_if.py
```
#### 2.3.2 TD3
```
python test_td3.py
```



## 3. Appendix
Inner state training logic:
```python
result = model(obses, next_obses)
error_e = F.mse_loss(model.get_keypoint(result['reconstructed_image_b'])["centers"], result['keypoints_b']["centers"])
decoder_error_e = F.mse_loss(result['reconstructed_image_b'], next_obses)
```
where, the structure of the result (which also named model.forward) is 
```python
return {
    "reconstructed_image_b": reconstructed_image_b,
    "features_a": image_a_features,
    "features_b": image_b_features,
    "keypoints_a": image_a_keypoints,
    "keypoints_b": image_b_keypoints,
}
```


