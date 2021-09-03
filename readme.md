# Background
## 论文总结
### Task-agnostic self-modeling machines
* 学习目标：状态转移
* 环境：机械臂
* 状态： 关节角度
* 动作：角度命令

### Unsupervised Learning of Object Keypoints for Perception and Control 
* 学习目标：关键点
* 环境：gym atari
* 状态：像素图片
* 方法： Transporter 网络 (tf): 构建keypointmap，来重构下一帧

### Invariant Causal Prediction for Block MDPs 
* 学习目标：因果预测
* 环境：噪声环境（自建）
* 方法：MISA （模型无关抽象） (th)：构建转移模型，通过隐层转移模型

### Reinforcement Learning through Active Inference


## 整合 
### Keypoints & Block MDPs
* 学习目标：状态表示
* 学习方法：建议状态转移预测网络，将ICP中 phi-model-decoder 三个网络 用Transporter部件重写
其中model部分变成keypointmap算式

### 状态表示 & Active Inference
* 学习目标：策略
* 学习方法：在经验池和planner中使用keypoint预处理state，转变为keypointmap center，再使用Active Inference逻辑进行运算和训练

# Environment
安装相关环境
```
pip install -r requirements.txt
```
也可以在运行时缺什么装什么

# Train
##观察训练情况
```python
tensorboard --logdir=./log
```

## 整合版本
```
 python main.py --num_samples 10 --num_iters 10 --batch_size 2 --log_interval 5 --replay_buffer_capacity 100 
 ```


## 学习状态表示
检查是否正常运行，这仅将运行很少的轮次做运行验证
```
 python train_predicter_collect_by_ddpg.py --num_samples 10 --num_iters 10 --batch_size 2 --log_interval 5 --replay_buffer_capacity 100  --num_envs 1
 ```
正式运行
如果机器足够强 可以直接运行 train_predicter.py 
也可以调整到其他恰当的参数。

### 随机采样

```
 python train_predicter_collect_by_ddpg.py --mode random --num_samples 10 --num_iters 10 --batch_size 2 --log_interval 5 --replay_buffer_capacity 100  --num_envs 1
```

### DDPG采样
单独训练ddpg 可以通过运行`train_ddpg.py`
使用一个已有策略采样
```
 python train_predicter_collect_by_ddpg.py --mode load --num_samples 10 --num_iters 10 --batch_size 2 --log_interval 5 --replay_buffer_capacity 100  --num_envs 1
 ```
使用重新训练策略采样
```
 python train_predicter_collect_by_ddpg.py --mode train --num_samples 10 --num_iters 10 --batch_size 2 --log_interval 5 --replay_buffer_capacity 100  --num_envs 1
 ```

## 训练策略
### IF
```
python train_if.py
```
正式运行
如果机器足够强 可以直接运行 test_td3.py
也可以调整到其他恰当的参数。

## TD3
```
python test_td3.py --batch-size 4 --training-num 2 --step-per-epoch 20
```
正式运行
如果机器足够强 可以直接运行 test_td3.py
也可以调整到其他恰当的参数。



## 参数说明

* 学习状态表示部分参数， 见main.parse_args()
* 学习策略部分参数， seed，batch_size，buffer_size 见main.parse_args()， 其他见pmbrl.get_config()
* 环境相关参数 在env对应环境中

# 代码逻辑说明
- 本项目的整合方式是把keypoint的网络整合到block-MDP的运算逻辑中，由于第一个论文的观点可以看作是block-MDP的化简版本因此没有采纳此文章代码。
- 从具体代码的角度看， block-MDP 的主要贡献为
dmc2gym和local_dm_control_suit两个文件夹，以及train_predicter的主体。
- keypoint的主要贡献为models文件夹中除了net的其他网络，注意原文使用了tensorflow，此处是使用pytorch重构的版本
- 最后在train_predicter中， block-MDP所设计的phi-model-decoder 三个网络被keypoint的transporter网络替代。
- block-MDP原网络和transporter网络的区别，原网络中phi-model-decoder分别代表，生成隐藏状态的编码器，从当前隐藏状态求解下一步隐藏状态的网络，从隐藏状态还原成原本图像的编码器。
transporter网络也包括编码器，和解码器，但没有下一步预测网络，而是加入keypoint网络，通过两帧的keypoint加上帧a的中间特征生成帧b的中间特征。由于我们的目标是使用keypoint作为隐藏状态，所以最终通过了一些转换实现了对应目标
##原本逻辑
```python
latent = phi(obses)
pred_next_latent = model(latent, actions)
true_next_latent = phi(next_obses).detach()
error_e = F.mse_loss(pred_next_latent, true_next_latent)
decoder_error_e = F.mse_loss(pred_next_obses, next_obses)
```
##转换逻辑
```python
result = model(obses, next_obses)
error_e = F.mse_loss(model.get_keypoint(result['reconstructed_image_b'])["centers"], result['keypoints_b']["centers"])
decoder_error_e = F.mse_loss(result['reconstructed_image_b'], next_obses)
```
需要额外说明的是result结构为（也就是model.forward返回）：
```python
return {
    "reconstructed_image_b": reconstructed_image_b,
    "features_a": image_a_features,
    "features_b": image_b_features,
    "keypoints_a": image_a_keypoints,
    "keypoints_b": image_b_keypoints,
}
```