import unittest

import numpy as np
import torch

from trinity.algorithm.advantage_fn import ADVANTAGE_FN
from trinity.common.experience import EID, Experience


class TestGroupedAdvantageFn(unittest.TestCase):
    """Test cases for group-based advantage functions."""

    def test_grpo_advantage(self):
        advantage_fn_cls = ADVANTAGE_FN.get("grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(**advantage_fn_cls.default_args())
        task_num = 3
        repeat_times = 5
        exps = [
            Experience(
                eid=EID(batch=0, task=j, run=i),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=i,
            )
            for i in range(repeat_times)
            for j in range(task_num)
        ]

        # test group_epxeriences
        grouped_exps = advantage_fn.group_experiences(exps)
        self.assertEqual(len(grouped_exps), task_num)

        # test calculate_group_advantage
        for group_id, group_exps in grouped_exps.items():
            modified_exps, group_metrics = advantage_fn.calculate_group_advantage(
                group_id, group_exps
            )
            self.assertEqual(len(modified_exps), repeat_times)
            self.assertIn("reward_mean", group_metrics)
            self.assertIn("reward_std", group_metrics)

        # test the full pipeline

        exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                ),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=i,
            )
            for i in range(repeat_times)
            for j in range(task_num)
        ]
        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), task_num * repeat_times)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertTrue(metrics["group_advantages/reward_mean/mean"] == 2.0)
        self.assertTrue(
            metrics["group_advantages/reward_std/mean"]
            == torch.std(torch.tensor([i for i in range(repeat_times)], dtype=torch.float32)).item()
        )

        repeat_times = 1
        exps = [
            Experience(
                eid=EID(batch=0, task=j, run=i),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=i,
            )
            for i in range(repeat_times)
            for j in range(task_num)
        ]
        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), task_num * repeat_times)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertTrue(metrics["group_advantages/reward_mean/mean"] == 0.0)
        self.assertTrue(metrics["group_advantages/reward_std/mean"] == 1.0)

    def test_grpo_reward_std(self):
        advantage_fn_cls = ADVANTAGE_FN.get("grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(epsilon=1e-6, std_threshold=0.0)
        task_num = 3
        repeat_times = 5
        exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                ),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=0.5,
            )
            for i in range(repeat_times)
            for j in range(task_num)
        ]

        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), 0)
        self.assertIn("filtered_count", metrics)
        self.assertEqual(metrics["filtered_count"], 15)

    def test_grpo_correct_bias(self):
        advantage_fn_cls = ADVANTAGE_FN.get("grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(epsilon=1e-7, rank_penalty=0.2)
        task_num = 2
        repeat_times = 4
        exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                ),
                tokens=torch.zeros(5),
                logprobs=torch.tensor([0.1 * i for _ in range(5)]),
                prompt_length=2,
                reward=i,
            )
            for i in range(repeat_times)
            for j in range(task_num)
        ]
        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), task_num * repeat_times)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertAlmostEqual(
            metrics["group_advantages/reward_mean/mean"],
            torch.mean(torch.tensor([0.0, 0.95, 1.80, 2.55], dtype=torch.float32)).item(),
            places=6,
        )
        self.assertAlmostEqual(
            metrics["group_advantages/reward_std/mean"],
            torch.std(torch.tensor([0.0, 0.95, 1.80, 2.55], dtype=torch.float32)).item(),
            places=6,
        )

    def test_batch_level_std_grpo(self):
        advantage_fn_cls = ADVANTAGE_FN.get("grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(epsilon=1e-7, std_cal_level="batch")

        rewards_task0 = [1.0, 2.0, 3.0]
        rewards_task1 = [11.0, 12.0, 13.0]

        exps = [
            Experience(
                eid=EID(batch=0, task=0, run=i),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=rewards_task0[i],
                action_mask=torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32),
            )
            for i in range(len(rewards_task0))
        ]
        exps.extend(
            [
                Experience(
                    eid=EID(batch=0, task=1, run=i),
                    tokens=torch.zeros(5),
                    prompt_length=2,
                    reward=rewards_task1[i],
                    action_mask=torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32),
                )
                for i in range(len(rewards_task1))
            ]
        )

        all_rewards = torch.tensor(rewards_task0 + rewards_task1, dtype=torch.float32)
        batch_std = torch.std(all_rewards)

        group0_mean = torch.mean(torch.tensor(rewards_task0, dtype=torch.float32))

        processed_exps, metrics = advantage_fn(exps)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertEqual(len(processed_exps), len(rewards_task0) + len(rewards_task1))

        target_exp = next(exp for exp in processed_exps if exp.eid.task == 0 and exp.eid.run == 1)
        expected_advantage_value = (target_exp.reward - group0_mean) / (
            batch_std + advantage_fn.epsilon
        )
        expected_advantages = expected_advantage_value * target_exp.action_mask
        self.assertTrue(torch.allclose(target_exp.advantages, expected_advantages, atol=1e-6))
        self.assertTrue(torch.allclose(target_exp.returns, expected_advantages, atol=1e-6))

    def test_duplicate_grpo(self):
        advantage_fn_cls = ADVANTAGE_FN.get("grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(epsilon=1e-6, std_threshold=0.0, duplicate_experiences=True)
        task_num = 3
        repeat_times = 5
        exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                ),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=np.random.rand(),
            )
            for i in range(repeat_times)
            for j in range(task_num - 1)
        ]
        zero_adv_exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                ),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=0.5,
            )
            for i in range(repeat_times)
            for j in range(task_num - 1, task_num * 2)
        ]
        exps.extend(zero_adv_exps)

        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), 2 * task_num * repeat_times)

        exps, metrics = advantage_fn(zero_adv_exps)
        self.assertEqual(len(exps), 0)

    def test_step_wise_grpo_advantage(self):
        advantage_fn_cls = ADVANTAGE_FN.get("step_wise_grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(epsilon=1e-7)
        self.assertEqual(advantage_fn.epsilon, 1e-7)
        task_num = 2
        repeat_times = 3
        step_num = 4
        exps = [
            Experience(
                eid=EID(
                    batch=0,
                    task=j,
                    run=i,
                    step=k,
                ),
                tokens=torch.zeros(5),
                prompt_length=2,
                reward=i,
            )
            for k in range(step_num)
            for i in range(repeat_times)
            for j in range(task_num)
        ]

        exps, metrics = advantage_fn(exps)
        self.assertEqual(len(exps), task_num * repeat_times * step_num)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertTrue(metrics["group_advantages/reward_mean/mean"] == 1.0)
        self.assertTrue(
            metrics["group_advantages/reward_std/mean"]
            == torch.std(torch.tensor([i for i in range(repeat_times)], dtype=torch.float32)).item()
        )

    def test_batch_level_step_wise_grpo_advantage(self):
        advantage_fn_cls = ADVANTAGE_FN.get("step_wise_grpo")
        self.assertIsNotNone(advantage_fn_cls)
        advantage_fn = advantage_fn_cls(epsilon=1e-7, std_cal_level="batch")

        task_num = 2
        repeat_times = 3  # runs
        step_num = 4

        # Let reward vary by task, run, and step to make the test meaningful
        # reward = task*10 + run*1 + step*0.1
        exps = []
        all_rewards_list = []
        for j in range(task_num):  # task
            for i in range(repeat_times):  # run
                reward_val = float(j * 10 + i * 1)
                all_rewards_list.append(reward_val)
                for k in range(step_num):  # step
                    exps.append(
                        Experience(
                            eid=EID(batch=0, task=j, run=i, step=k),
                            tokens=torch.zeros(5),
                            prompt_length=2,
                            reward=reward_val,
                            action_mask=torch.tensor([0, 0, 1, 1, 1], dtype=torch.float32),
                        )
                    )

        all_rewards = torch.tensor(all_rewards_list, dtype=torch.float32)
        batch_std = torch.std(all_rewards)

        # For a specific group (e.g., task = 9)
        group_rewards = [
            float(0 * 10 + 1 * k) for k in range(repeat_times)
        ]  # [0.0, 1.0, 2.0] for task = 0
        group_mean = torch.mean(torch.tensor(group_rewards, dtype=torch.float32))

        processed_exps, metrics = advantage_fn(exps)
        self.assertIn("group_advantages/reward_mean/mean", metrics)
        self.assertIn("group_advantages/reward_std/mean", metrics)
        self.assertEqual(len(processed_exps), task_num * repeat_times * step_num)

        # Pick a target experience: task=0, run=1, step=2. Reward is 1.2
        target_exp = next(
            exp
            for exp in processed_exps
            if exp.eid.task == 0 and exp.eid.run == 1 and exp.eid.step == 0
        )
        expected_advantage_value = (target_exp.reward - group_mean) / (
            batch_std + advantage_fn.epsilon
        )
        expected_advantages = expected_advantage_value * target_exp.action_mask
        self.assertTrue(torch.allclose(target_exp.advantages, expected_advantages, atol=1e-6))
        self.assertTrue(torch.allclose(target_exp.returns, expected_advantages, atol=1e-6))
