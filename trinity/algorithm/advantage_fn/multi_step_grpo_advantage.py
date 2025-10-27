"""GRPO advantage computation for multi-step scenarios
"""
from typing import Dict, List, Optional, Tuple

import torch

from trinity.algorithm.advantage_fn.advantage_fn import ADVANTAGE_FN, AdvantageFn
from trinity.buffer.operators import ExperienceOperator
from trinity.common.experience import Experience, group_by
from trinity.utils.monitor import gather_metrics


@ADVANTAGE_FN.register_module("step_wise_grpo")
class StepWiseGRPOAdvantageFn(AdvantageFn, ExperienceOperator):
    """
    An advantage function that broadcasts advantages from the last step to previous steps.
    Inspired by rLLM (https://github.com/rllm-org/rllm).
    """

    def __init__(
        self,
        epsilon: float = 1e-6,
        enable_step_norm: bool = False,
        std_cal_level: str = "group",  # 'group' (task-level) or 'batch'
        **kwargs,
    ) -> None:
        """Initialize the Step-wise GRPO advantage function.

        Args:
            epsilon (float): A small value to avoid division by zero.
            enable_step_norm (bool): If True, normalize advantages by trajectory length.
            std_cal_level (str): The scope for calculating reward standard deviation.
                'group' (default): Std is calculated per task group.
                'batch': Std is calculated across all last-step rewards in the entire batch.
                The mean is always calculated per task group.
        """
        self.epsilon = epsilon
        self.enable_step_norm = enable_step_norm
        self.std_cal_level = std_cal_level
        if self.std_cal_level not in ["group", "batch"]:
            raise ValueError("std_cal_level must be either 'group' or 'batch'")

    def calculate_last_step_advantage(
        self,
        exps: Dict[str, Experience],
        precomputed_std: Optional[torch.Tensor] = None,
    ) -> Tuple[Dict[str, float], Dict[str, float]]:
        """Calculate group advantage for a given group of experiences.

        Args:
            exps (Dict[str, Experience]): One experience per run, keyed by run ID.

        Returns:
            Dict[str, float]: A tuple containing the scores for each run.
            Dict[str, float]: Metrics for logging.
        """
        with torch.no_grad():
            if len(exps) == 1:
                group_reward_mean = torch.tensor(0.0)
                group_reward_std = torch.tensor(1.0)
            else:
                rewards = torch.tensor([exp.reward for exp in exps.values()], dtype=torch.float32)
                group_reward_mean = torch.mean(rewards)
                group_reward_std = torch.std(rewards)
            scores = {}
            for rid, exp in exps.items():
                if self.std_cal_level == "batch" and precomputed_std is not None:
                    score = (exp.reward - group_reward_mean) / (precomputed_std + self.epsilon)
                else:
                    score = (exp.reward - group_reward_mean) / (group_reward_std + self.epsilon)
                scores[rid] = score.item()
            metrics = {
                "reward_mean": group_reward_mean.item(),
                "reward_std": group_reward_std.item(),
            }
        return scores, metrics

    def broadcast_advantages(
        self, run_exps: Dict[str, List[Experience]], scores: Dict[str, float]
    ) -> Dict[str, List[Experience]]:
        """Broadcast the calculated advantages to all previous steps in each run.

        Args:
            run_exps (Dict[str, List[Experience]]): Experiences grouped by run ID.
            scores (Dict[str, float]): Calculated scores for each run.

        Returns:
            Dict[str, List[Experience]]: Updated experiences with advantages broadcasted.
        """
        for run_id, exps in run_exps.items():
            score = scores[run_id]
            traj_length = len(exps)
            for exp in exps:
                exp.advantages = exp.action_mask * score  # type: ignore [operator]
                if self.enable_step_norm:
                    exp.advantages /= traj_length
                exp.returns = exp.advantages.clone()
        return run_exps

    def process(self, exps: List[Experience]) -> Tuple[List[Experience], Dict]:
        if len(exps) == 0:
            return [], {}
        cnt = 0
        metric_list = []
        # Step 1: split the experiences into sub-groups by task
        task_exps = group_by(exps, "task")

        # --- Pre-computation step for batch-level standard deviation ---
        precomputed_std = None
        if self.std_cal_level == "batch":
            all_laststep_rewards = []
            for task_exp in task_exps.values():
                # First, group all experiences by run to find the last step of each run
                task_run_exps = group_by(task_exp, "run")
                # Collect rewards from the last step of every run in the entire batch
                last_step_rewards = [
                    run_steps[-1].reward for run_steps in task_run_exps.values() if run_steps
                ]
                all_laststep_rewards.extend(last_step_rewards)

            if len(all_laststep_rewards) <= 1:
                precomputed_std = torch.tensor(1.0)
            else:
                precomputed_std = torch.std(torch.tensor(all_laststep_rewards, dtype=torch.float32))
        # --- End of pre-computation ---

        # Step 2: further split each task's experiences into sub-groups by run
        result_exps = []
        for task_exp in task_exps.values():
            run_exps = group_by(task_exp, "run")

            # Step3: extract the last experience (last step) from each run and calculate scores
            last_step_exps = {run_id: step_exps[-1] for run_id, step_exps in run_exps.items()}
            scores, metrics = self.calculate_last_step_advantage(
                last_step_exps, precomputed_std=precomputed_std
            )
            metric_list.append(metrics)

            # Step 4: broadcast the advantages to all previous steps
            run_exps = self.broadcast_advantages(run_exps, scores)
            for exps in run_exps.values():
                cnt += len(exps)
                result_exps.extend(exps)

        metrics = gather_metrics(metric_list, "group_advantages")
        metrics["experience_count"] = cnt
        return result_exps, metrics

    def __call__(self, exps, **kwargs):
        return self.process(exps)

    @classmethod
    def compute_in_trainer(cls) -> bool:
        """Whether the advantage should be computed in the trainer loop."""
        return False

    @classmethod
    def default_args(cls) -> Dict:
        """Return the default configuration for this strategy."""
        return {
            "epsilon": 1e-6,
            "enable_step_norm": False,
        }
