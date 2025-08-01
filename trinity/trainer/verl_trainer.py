# -*- coding: utf-8 -*-
"""veRL Trainer Class

Modified from verl/trainer/ppo/ray_trainer.py
"""
import os
import sys
from typing import Dict, Tuple

import ray
import torch
from omegaconf import OmegaConf
from verl.trainer.ppo.metric_utils import (
    compute_throughout_metrics,
    compute_timing_metrics,
    reduce_metrics,
)
from verl.trainer.ppo.ray_trainer import (
    RayClassWithInitArgs,
    RayPPOTrainer,
    RayWorkerGroup,
    ResourcePoolManager,
    Role,
    create_colocated_worker_cls,
    find_latest_ckpt_path,
)
from verl.utils import hf_tokenizer
from verl.utils.debug import marked_timer
from verl.utils.fs import copy_local_path_from_hdfs

from trinity.algorithm import ADVANTAGE_FN, KL_FN, SAMPLE_STRATEGY
from trinity.algorithm.algorithm import ALGORITHM_TYPE, SFTAlgorithm
from trinity.algorithm.algorithm_manager import AlgorithmManager
from trinity.algorithm.utils import prefix_metrics
from trinity.common.config import Config
from trinity.common.experience import Experiences
from trinity.trainer.trainer import TrainEngineWrapper
from trinity.trainer.verl.utils import compute_data_metrics, to_data_proto
from trinity.utils.log import get_logger


class _InternalDataLoader:
    def __init__(self, config):
        self.config = config
        self.dataset = None
        self.index = 0
        self.experience_buffer = None

    def state_dict(self):
        return None

    def load_state_dict(self, *args, **kwargs):
        pass

    def __getstate__(self):
        state = self.__dict__.copy()
        return state

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __iter__(self):
        self.index = 0
        return self

    def __next__(self):
        raise StopIteration


class VerlPPOTrainerWrapper(RayPPOTrainer, TrainEngineWrapper):
    """A wrapper for verl.trainer.ppo.RayPPOTrainer."""

    def __init__(
        self,
        global_config: Config,
    ):
        train_config = global_config.trainer
        config = OmegaConf.structured(train_config.trainer_config)
        # download the checkpoint from hdfs
        local_path = copy_local_path_from_hdfs(config.actor_rollout_ref.model.path)

        # instantiate tokenizer

        tokenizer = hf_tokenizer(local_path)

        # define worker classes
        if config.actor_rollout_ref.actor.strategy == "fsdp":
            assert config.actor_rollout_ref.actor.strategy == config.critic.strategy
            from trinity.trainer.verl.fsdp_workers import (
                ActorRolloutRefWorker,
                CriticWorker,
            )

            ray_worker_group_cls = RayWorkerGroup

        elif config.actor_rollout_ref.actor.strategy == "megatron":
            raise NotImplementedError("Not support megatron for now.")

        else:
            raise NotImplementedError

        role_worker_mapping = {
            Role.ActorRollout: ray.remote(ActorRolloutRefWorker),
            Role.Critic: ray.remote(CriticWorker),
            Role.RefPolicy: ray.remote(ActorRolloutRefWorker),
        }

        global_pool_id = "global_pool"
        resource_pool_spec = {
            global_pool_id: [config.trainer.n_gpus_per_node] * config.trainer.nnodes,
        }
        mapping = {
            Role.ActorRollout: global_pool_id,
            Role.Critic: global_pool_id,
            Role.RefPolicy: global_pool_id,
        }

        resource_pool_manager = ResourcePoolManager(
            resource_pool_spec=resource_pool_spec, mapping=mapping
        )
        self.algorithm_config = global_config.algorithm
        self.algorithm = None
        self.algorithm_manager = AlgorithmManager(global_config)

        # specify advantage function for various rft algorithms
        algorithm = ALGORITHM_TYPE.get(self.algorithm_config.algorithm_type)
        if algorithm.compute_advantage_in_trainer:
            self.advantage_fn = ADVANTAGE_FN.get(self.algorithm_config.advantage_fn)(
                **self.algorithm_config.advantage_fn_args
            )
            self.kl_fn = KL_FN.get(self.algorithm_config.kl_penalty_fn)(
                **self.algorithm_config.kl_penalty_fn_args
            )
        self.sample_strategy = SAMPLE_STRATEGY.get(global_config.algorithm.sample_strategy)(
            buffer_config=global_config.buffer,
            trainer_type=global_config.trainer.trainer_type,
            **global_config.algorithm.sample_strategy_args,
        )
        super().__init__(
            config,
            tokenizer,
            role_worker_mapping,
            resource_pool_manager,
            ray_worker_group_cls,
        )
        self.init_workers()
        self.logger = get_logger(__name__)
        self.last_full_save_step = None

    def _validate_config(self):  # TODO
        algorithm = ALGORITHM_TYPE.get(self.algorithm_config.algorithm_type)
        self.use_critic = algorithm.use_critic
        super()._validate_config()

    def init_workers(self):
        """Initialize distributed training workers using Ray backend.


        Creates:

        1. Ray resource pools from configuration

        2. Worker groups for each role (actor, critic, etc.)

        """
        self.resource_pool_manager.create_resource_pool()

        self.resource_pool_to_cls = {
            pool: {} for pool in self.resource_pool_manager.resource_pool_dict.values()
        }

        # create actor and rollout
        if self.hybrid_engine:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.ActorRollout)
            actor_rollout_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.ActorRollout],
                config=self.config.actor_rollout_ref,
                role="actor",
            )
            self.resource_pool_to_cls[resource_pool]["actor"] = actor_rollout_cls
        else:
            raise NotImplementedError

        # create critic
        if self.use_critic:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.Critic)
            critic_cls = RayClassWithInitArgs(
                cls=self.role_worker_mapping[Role.Critic], config=self.config.critic
            )
            self.resource_pool_to_cls[resource_pool]["critic"] = critic_cls

        # create reference policy if needed
        if self.use_reference_policy:
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RefPolicy)
            ref_policy_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RefPolicy],
                config=self.config.actor_rollout_ref,
                role="ref",
            )
            self.resource_pool_to_cls[resource_pool]["ref"] = ref_policy_cls

        # create a reward model if reward_fn is None
        if self.use_rm:
            # we create a RM here
            resource_pool = self.resource_pool_manager.get_resource_pool(Role.RewardModel)
            rm_cls = RayClassWithInitArgs(
                self.role_worker_mapping[Role.RewardModel], config=self.config.reward_model
            )
            self.resource_pool_to_cls[resource_pool]["rm"] = rm_cls

        # initialize WorkerGroup
        # NOTE: if you want to use a different resource pool for each role, which can support different parallel size,
        # you should not use `create_colocated_worker_cls`.
        # Instead, directly pass different resource pool to different worker groups.
        # See https://github.com/volcengine/verl/blob/master/examples/ray/tutorial.ipynb for more information.
        all_wg = {}
        wg_kwargs = {}  # Setting up kwargs for RayWorkerGroup
        if OmegaConf.select(self.config.trainer, "ray_wait_register_center_timeout") is not None:
            wg_kwargs[
                "ray_wait_register_center_timeout"
            ] = self.config.trainer.ray_wait_register_center_timeout
        for resource_pool, class_dict in self.resource_pool_to_cls.items():
            worker_dict_cls = create_colocated_worker_cls(class_dict=class_dict)
            wg_dict = self.ray_worker_group_cls(
                resource_pool=resource_pool,
                ray_cls_with_init=worker_dict_cls,
                device_name=self.device_name,
                **wg_kwargs,
            )
            spawn_wg = wg_dict.spawn(prefix_set=class_dict.keys())
            all_wg.update(spawn_wg)

        if self.use_critic:
            self.critic_wg = all_wg["critic"]
            self.critic_wg.init_model()

        if self.use_reference_policy and not self.ref_in_actor:
            self.ref_policy_wg = all_wg["ref"]
            self.ref_policy_wg.init_model()

        if self.use_rm:
            self.rm_wg = all_wg["rm"]
            self.rm_wg.init_model()

        # we should create rollout at the end so that vllm can have a better estimation of kv cache memory
        self.actor_rollout_wg = all_wg["actor"]
        self.actor_rollout_wg.init_model()

    @property
    def train_step_num(self) -> int:
        return self.global_steps

    def prepare(self):
        self.actor_rollout_wg.setup_weight_sync_group()

        # The global step counter, initialized to 0
        # It represents the total number of training steps completed so far
        # We increment this counter at the beginning of each training step
        self.global_steps = 0

        # load checkpoint before doing anything
        self._load_checkpoint()

    def _create_dataloader(self, train_dataset, val_dataset, collate_fn, train_sampler):
        self.train_dataloader = _InternalDataLoader(self.config)
        # TODO: compute total training steps
        self.total_training_steps = self.config.trainer.total_training_steps or sys.maxsize

    def save_state_dict(self):  # checkpoint sync
        local_global_step_folder = os.path.join(
            self.config.trainer.default_local_dir, f"global_step_{self.global_steps}"
        )

        actor_local_path = os.path.join(local_global_step_folder, "actor")
        if not os.path.exists(actor_local_path):
            self.actor_rollout_wg.save_checkpoint(
                actor_local_path,
                None,
                self.global_steps,
                max_ckpt_to_keep=None,
                model_state_dict_only=True,
            )

    def upload_state_dict(self):  # state dict sync
        self.actor_rollout_wg.upload_state_dict(self.global_steps)

    def train_step(self, batch: Experiences) -> Tuple[bool, Dict]:  # noqa C901
        self.logger.info(f"Training at step {self.global_steps + 1} started.")
        batch = to_data_proto(batch)
        metrics = {}
        self.global_steps += 1
        self.logger.info(f"Sampling at step {self.global_steps} done.")
        timing_raw = {}
        algorithm_config = self.algorithm_manager.get_current_algorithm_config(self.global_steps)
        algorithm = ALGORITHM_TYPE.get(algorithm_config.algorithm_type)
        if self.algorithm != algorithm:
            self.actor_rollout_wg.set_algorithm(algorithm_config)
            if self.algorithm == SFTAlgorithm:
                self.sft_to_rft()
            self.algorithm = algorithm

        with marked_timer("step", timing_raw):
            batch.meta_info["temperature"] = self.config.actor_rollout_ref.rollout.temperature

            if self.algorithm.can_balance_batch and self.config.trainer.balance_batch:
                self._balance_batch(batch, metrics=metrics)  # TODO this may affect multi-turn

            # compute global_valid tokens
            batch.meta_info["global_token_num"] = torch.sum(
                batch.batch["attention_mask"], dim=-1
            ).tolist()

            if self.algorithm.use_reference:  # ref_logprob may not be used
                # compute reference log_prob
                with marked_timer("ref", timing_raw):
                    ref_log_prob = self.ref_policy_wg.compute_ref_log_prob(batch)
                    batch = batch.union(ref_log_prob)

            if self.algorithm.use_critic:
                with marked_timer("values", timing_raw):
                    values = self.critic_wg.compute_values(batch)
                    batch = batch.union(values)

            if self.algorithm.compute_advantage_in_trainer:
                with marked_timer("adv", timing_raw):
                    # compute kl penalty
                    batch, kl_metrics = self.kl_fn.apply_kl_penalty_to_reward(batch)
                    metrics.update(prefix_metrics(kl_metrics, prefix="critic"))
                    # compute advantages, executed on the driver process
                    batch, _ = self.advantage_fn(batch)

                # update critic
            if self.algorithm.use_critic:
                with marked_timer("update_critic", timing_raw):
                    critic_output = self.critic_wg.update_critic(batch)
                critic_output_metrics = reduce_metrics(critic_output.meta_info["metrics"])
                metrics.update(critic_output_metrics)

            # implement critic warmup
            if (
                not self.algorithm.use_critic
                or self.config.trainer.critic_warmup <= self.global_steps
            ):
                # update actor
                with marked_timer("update_actor", timing_raw):
                    actor_output = self.actor_rollout_wg.update_actor(batch)
                    # TODO add send weight explorer
                actor_output_metrics = reduce_metrics(actor_output.meta_info["metrics"])
                metrics.update(actor_output_metrics)

        # collect metrics
        metrics.update(compute_data_metrics(batch=batch, use_critic=self.use_critic))
        metrics.update(compute_timing_metrics(batch=batch, timing_raw=timing_raw))
        n_gpus = self.resource_pool_manager.get_n_gpus()
        metrics.update(
            compute_throughout_metrics(batch=batch, timing_raw=timing_raw, n_gpus=n_gpus)
        )

        train_status = self.global_steps < self.total_training_steps
        if (
            not train_status
            or self.algorithm_manager.need_save(self.global_steps)
            or (
                self.config.trainer.save_freq > 0
                and self.global_steps % self.config.trainer.save_freq == 0
            )
        ):
            self.logger.info(f"Saving at step {self.global_steps}.")
            with marked_timer("save_checkpoint", timing_raw):
                self.save_checkpoint()
            self.logger.info(f"Saved at step {self.global_steps}.")
        self.logger.info(f"Training at step {self.global_steps} finished.")
        return train_status, metrics

    def save_checkpoint(self, block_until_saved: bool = False) -> None:
        if self.last_full_save_step != self.global_steps:
            self.last_full_save_step = self.global_steps
            self._save_checkpoint()
        if block_until_saved:
            self.actor_rollout_wg.wait_on_save_thread()
            if self.algorithm.use_critic:
                self.critic_wg.wait_on_save_thread()

    def sync_weight(self) -> None:
        self.actor_rollout_wg.sync_weight()

    def sft_to_rft(self) -> None:
        # load from hdfs
        if self.config.trainer.default_hdfs_dir is not None:
            raise NotImplementedError("load from hdfs is not implemented yet")
        else:
            checkpoint_folder = self.config.trainer.default_local_dir  # TODO: check path
            if not os.path.isabs(checkpoint_folder):
                working_dir = os.getcwd()
                checkpoint_folder = os.path.join(working_dir, checkpoint_folder)
            global_step_folder = find_latest_ckpt_path(checkpoint_folder)  # None if no latest

        # find global_step_folder
        self.actor_rollout_wg.wait_on_save_thread()
        if self.config.trainer.resume_mode == "auto":
            if global_step_folder is None:
                self.logger.info("Training from scratch")
                return
        else:
            if not (self.config.trainer.resume_from_path and global_step_folder is not None):
                assert isinstance(
                    self.config.trainer.resume_mode, str
                ), "resume ckpt must be str type"
                assert (
                    "global_step_" in self.config.trainer.resume_mode
                ), "resume ckpt must specify the global_steps"
                global_step_folder = self.config.trainer.resume_mode
                if not os.path.isabs(global_step_folder):
                    working_dir = os.getcwd()
                    global_step_folder = os.path.join(working_dir, global_step_folder)
        self.logger.info(f"Load from checkpoint folder: {global_step_folder}")
        # set global step
        global_steps = int(global_step_folder.split("global_step_")[-1])
        assert self.global_steps == global_steps + 1

        self.logger.info(f"Resuming from {global_step_folder}")

        actor_path = os.path.join(global_step_folder, "actor")
        self.logger.info(f"Loading actor from {actor_path} to ref_policy_wg")
        self.ref_policy_wg.load_checkpoint(actor_path, del_local_after_load=False)
        self.actor_rollout_wg.clear_optimizer_state()
        if self.use_critic:
            self.critic_wg.clear_optimizer_state()
        self.logger.info("sft to rft finished")
