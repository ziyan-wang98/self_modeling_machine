import argparse
import json
import os

import train_predicter_collect_by_ddpg
import train_if
from pmbrl import get_config


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--alpha", default=0.5, type=float, help="model error : decode error")
    parser.add_argument("--mode", default='random', type=str, help="collect mode random, load policy, train new policy")


    # environment
    parser.add_argument("--domain_name", default="cheetah")
    parser.add_argument("--task_name", default="run")
    parser.add_argument("--image_size", default=48, type=int)
    parser.add_argument("--action_repeat", default=1, type=int)
    parser.add_argument("--frame_stack", default=3, type=int)
    parser.add_argument("--num_envs", default=2, type=int)
    # replay buffer
    parser.add_argument("--replay_buffer_capacity", default=1000000, type=int)
    parser.add_argument("--num_samples", default=50000, type=int)
    # training
    parser.add_argument("--num_iters", default=100000, type=int)
    parser.add_argument("--lr", default=1e-3, type=float)
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument(
        "--one_decoder", action="store_true", help="baseline with single decoder"
    )
    # keypoint
    parser.add_argument("--num_keypoints", default=5, type=int)
    parser.add_argument("--gauss_std", default=0.1, type=float)

    # encoder/decoder
    parser.add_argument("--encoder_type", default="identity", type=str)
    parser.add_argument("--encoder_feature_dim", default=50, type=int)
    parser.add_argument("--encoder_lr", default=1e-3, type=float)
    parser.add_argument("--encoder_tau", default=0.05, type=float)
    parser.add_argument("--num_layers", default=4, type=int)
    parser.add_argument("--num_filters", default=32, type=int)
    # misc
    parser.add_argument("--seed", default=1, type=int)
    parser.add_argument("--log_interval", default=1000, type=int)
    parser.add_argument("--work_dir", default=".", type=str)
    parser.add_argument("--save_tb", default=False, action="store_true")
    parser.add_argument("--save_model", default=False, action="store_true")
    parser.add_argument("--save_buffer", default=False, action="store_true")
    parser.add_argument("--save_video", default=False, action="store_true")


    # if
    parser.add_argument("--logdir", type=str, default="log")
    parser.add_argument("--strategy", type=str, default="information")

    args = parser.parse_args()
    return args

if __name__ == "__main__":
    args = parse_args()
    print("train predicter", args)
    train_predicter.main(args)

    print("train policy")

    config = get_config(args)
    config.batch_size = args.batch_size
    config.buffer_size = args.replay_buffer_capacity
    train_if.main(config)


