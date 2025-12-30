"""
# @Time    : 2021/6/30 10:07 下午
# @Author  : hezhiqiang
# @Email   : tinyzqh@163.com
# @File    : train.py
"""

import os
import sys
from pathlib import Path

import numpy as np
import setproctitle
import torch

# Get the parent directory of the current file
parent_dir = os.path.abspath(os.path.join(os.getcwd(), "."))
# Append the parent directory to sys.path, otherwise the following import will fail
sys.path.append(parent_dir)

from config_mappo import get_config
from envs.env_wrappers import DummyVecEnv

NPU_IS_AVAILABLE = False
try:
    import torch_npu

    NPU_IS_AVAILABLE = True
except ImportError:
    print("torch_npu is not found.")


"""Train script for MPEs."""


def make_train_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.

            from envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv()

            # from envs.env_discrete import DiscreteActionEnv

            # env = DiscreteActionEnv()

            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def make_eval_env(all_args):
    def get_env_fn(rank):
        def init_env():
            # TODO 注意注意，这里选择连续还是离散可以选择注释上面两行，或者下面两行。
            # TODO Important, here you can choose continuous or discrete action space by uncommenting the above two lines or the below two lines.
            from envs.env_continuous import ContinuousActionEnv

            env = ContinuousActionEnv()
            # from envs.env_discrete import DiscreteActionEnv
            # env = DiscreteActionEnv()
            env.seed(all_args.seed + rank * 1000)
            return env

        return init_env

    return DummyVecEnv([get_env_fn(i) for i in range(all_args.n_rollout_threads)])


def parse_args(args, parser):
    parser.add_argument(
        "--scenario_name", type=str, default="MyEnv", help="Which scenario to run on"
    )
    parser.add_argument("--num_landmarks", type=int, default=3)
    parser.add_argument("--num_agents", type=int, default=2, help="number of players")

    all_args = parser.parse_known_args(args)[0]
    print("=" * 70)
    for k, v in all_args.__dict__.items():
        if v is not None:
            print(f"{k:<30} = {v:<20}")
    print("=" * 70)
    return all_args


def main(args):
    parser = get_config()
    all_args = parse_args(args, parser)

    if all_args.algorithm_name == "rmappo":
        assert all_args.use_recurrent_policy or all_args.use_naive_recurrent_policy, (
            "check recurrent policy!"
        )
    elif all_args.algorithm_name == "mappo":
        assert not all_args.use_recurrent_policy and not all_args.use_naive_recurrent_policy, (
            "check recurrent policy!"
        )
    else:
        raise NotImplementedError

    assert not (all_args.share_policy and all_args.scenario_name == "simple_speaker_listener"), (
        "The simple_speaker_listener scenario can not use shared policy. Please check the config.py."
    )

    device = "cpu"
    if all_args.cuda:
        if torch.cuda.is_available():
            device = "cuda:0"
        if NPU_IS_AVAILABLE:
            if torch_npu.npu.is_available():
                device = "npu:0"
                torch_npu.npu.set_compile_mode(jit_compile=False)
        if all_args.cuda_deterministic:
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
    device = torch.device(device)
    print(f"Choose device: {device}")
    torch.set_default_device(device)
    torch.set_num_threads(all_args.n_training_threads)
    # run dir
    run_dir = (
        Path(os.path.split(os.path.dirname(os.path.abspath(__file__)))[0] + "/results")
        / all_args.env_name
        / all_args.scenario_name
        / all_args.algorithm_name
        / all_args.experiment_name
    )
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)

    if not run_dir.exists():
        curr_run = "run1"
    else:
        exst_run_nums = [
            int(str(folder.name).split("run")[1])
            for folder in run_dir.iterdir()
            if str(folder.name).startswith("run")
        ]
        if len(exst_run_nums) == 0:
            curr_run = "run1"
        else:
            curr_run = "run%i" % (max(exst_run_nums) + 1)
    run_dir = run_dir / curr_run
    if not run_dir.exists():
        run_dir.mkdir(parents=True, exist_ok=True)

    setproctitle.setproctitle(
        str(all_args.algorithm_name)
        + "-"
        + str(all_args.env_name)
        + "-"
        + str(all_args.experiment_name)
        + "@"
        + str(all_args.user_name)
    )

    # seed
    np.random.seed(all_args.seed)
    torch.manual_seed(all_args.seed)
    torch.cuda.manual_seed_all(all_args.seed)
    torch_npu.npu.manual_seed_all(all_args.seed)

    # env init
    envs = make_train_env(all_args)
    eval_envs = make_eval_env(all_args) if all_args.use_eval else None
    num_agents = all_args.num_agents

    config = {
        "all_args": all_args,
        "envs": envs,
        "eval_envs": eval_envs,
        "num_agents": num_agents,
        "device": device,
        "run_dir": run_dir,
    }

    # run experiments
    if all_args.share_policy:
        from runner.shared.env_runner import EnvRunner as Runner
    else:
        from runner.separated.env_runner import EnvRunner as Runner

    runner = Runner(config)
    runner.run()

    # post process
    envs.close()
    if all_args.use_eval and eval_envs is not envs:
        eval_envs.close()

    runner.writter.export_scalars_to_json(str(runner.log_dir + "/summary.json"))
    runner.writter.close()


if __name__ == "__main__":
    main(sys.argv[1:])
