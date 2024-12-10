"""Pretrain GPT with dynamic parallelism support."""

import torch
from functools import partial
from megatron import get_args
from megatron import print_rank_0
from megatron import get_timers
from megatron import get_tokenizer
from megatron import mpu

from megatron.core.strategy_selector import DynamicStrategySelector
from megatron.core.model_profiler import ModelProfiler
from megatron.core.hardware_profiler import HardwareProfiler
from megatron.core.parallelism_manager import ParallelismManager

from megatron.data.gpt_dataset import build_train_valid_test_datasets, build_dataset_group
from megatron.enums import AttnMaskType
from megatron.model import GPTModel, GPTModelPipe
from megatron.training import pretrain
from megatron.utils import get_ltor_masks_and_position_ids, get_prefix_indices
from megatron.utils import average_losses_across_data_parallel_group

import deepspeed
from deepspeed.runtime.utils import see_memory_usage
import os

try:
    from torch.distributed.elastic.multiprocessing.errors import record
except ImportError:
    # noop
    def record(fn):
        return fn

def model_provider(pre_process=True, post_process=True):
    """Build the model with dynamic parallelism support."""

    print_rank_0('building GPT model ...')
    see_memory_usage(f"Before Building Model", force=True)

    args = get_args()

    with deepspeed.zero.Init(data_parallel_group=mpu.get_data_parallel_group(),
                             remote_device=None if args.remote_device == 'none' else args.remote_device,
                             config_dict_or_path=args.deepspeed_config,
                             enabled=args.zero_stage == 3,
                             mpu=mpu):
        if args.deepspeed:
            args.pretrain_causal_attention = True
            model = GPTModelPipe(
                num_tokentypes=0,
                parallel_output=True,
                attn_mask_type=AttnMaskType.causal
            )
            # Initialize dynamic parallelism components
            model_profiler = ModelProfiler(model)
            hardware_profiler = HardwareProfiler()
            parallelism_manager = ParallelismManager()

            # Initialize dynamic strategy selector
            model.strategy_selector = DynamicStrategySelector(
                model,
                model_profiler=model_profiler,
                hardware_profiler=hardware_profiler,
                parallelism_manager=parallelism_manager
            )
            
            # This is a hack to give us a reference to get_batch_pipe from within training.py
            model._megatron_batch_fn = get_batch_pipe
        else:
            model = GPTModel(
                num_tokentypes=0,
                parallel_output=True,
                attn_mask_type=AttnMaskType.causal,
                pre_process=pre_process,
                post_process=post_process
            )
            
            # Initialize dynamic parallelism components
            model_profiler = ModelProfiler(model)
            hardware_profiler = HardwareProfiler()
            parallelism_manager = ParallelismManager()

            # Initialize dynamic strategy selector
            model.strategy_selector = DynamicStrategySelector(
                model,
                model_profiler=model_profiler,
                hardware_profiler=hardware_profiler,
                parallelism_manager=parallelism_manager
            )
            
    see_memory_usage(f"After Building Model", force=True)
    return model


def get_batch(data_iterator):
    """Generate a batch"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    if data_iterator is not None:
        data = next(data_iterator)
    else:
        data = None
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and postition ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        prefix_indices=None,
        loss_on_targets_only=args.loss_on_targets_only
    )

    return tokens, labels, loss_mask, attention_mask, position_ids


def get_batch_pipe(data):
    """Modification of `get_batch` to work on `next(data_iterator)` instead of `data_iterator`"""
    args = get_args()
    tokenizer = get_tokenizer()

    # Items and their type.
    keys = ['text']
    datatype = torch.int64

    # Broadcast data.
    data_b = mpu.broadcast_data(keys, data, datatype)

    # Unpack.
    tokens_ = data_b['text'].long()
    labels = tokens_[:, 1:].contiguous()
    tokens = tokens_[:, :-1].contiguous()

    # Get the masks and position ids.
    attention_mask, loss_mask, position_ids = get_ltor_masks_and_position_ids(
        tokens,
        tokenizer.eod,
        args.reset_position_ids,
        args.reset_attention_mask,
        args.eod_mask_loss,
        prefix_indices=None,
        loss_on_targets_only=args.loss_on_targets_only
    )
    if args.curriculum_learning and args.curriculum_seqlen < tokens.size()[1]:
        # seqlen-based curriculum learning
        # tokens, position_ids, labels, loss_mask have size [batch size, seqlen]
        tokens = tokens[:, :args.curriculum_seqlen].contiguous()
        position_ids = position_ids[:, :args.curriculum_seqlen].contiguous()
        labels = labels[:, :args.curriculum_seqlen].contiguous()
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return (tokens, position_ids, attention_mask), (labels, loss_mask)


def loss_func(loss_mask, output_tensor):
    losses = output_tensor.float()
    loss_mask = loss_mask.view(-1).float()
    loss = torch.sum(losses.view(-1) * loss_mask) / loss_mask.sum()

    # Reduce loss for logging.
    averaged_loss = average_losses_across_data_parallel_group([loss])

    return loss, {'lm loss': averaged_loss[0]}


def forward_step(data_iterator, model):
    """Forward step."""
    args = get_args()
    timers = get_timers()

    # Get the batch.
    timers('batch-generator').start()
    tokens, labels, loss_mask, attention_mask, position_ids = get_batch(
        data_iterator)
    timers('batch-generator').stop()

    output_tensor = model(tokens, position_ids, attention_mask,
                          labels=labels)
    if args.curriculum_learning and args.curriculum_seqlen < args.seq_length:
        loss_mask = loss_mask[:, :args.curriculum_seqlen].contiguous()

    return output_tensor, partial(loss_func, loss_mask)


def train_step(forward_step_func, data_iterator, model, optimizer, lr_scheduler):
    """Single training step with dynamic parallelism support."""
    args = get_args()
    timers = get_timers()

    # Update strategy based on performance metrics
    if hasattr(model, 'strategy_selector'):
        metrics = {
            'throughput': args.global_batch_size / timers('interval-time').elapsed(),
            'gpu_memory_util': torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated(),
            'gpu_compute_util': timers('forward').elapsed() / timers('interval-time').elapsed()
        }
        model.strategy_selector.select_strategy(metrics)

    # Forward pass
    losses = forward_step_func(data_iterator, model)
    loss = losses['loss']
    
    # Backward pass
    backward_step(optimizer, model, loss)

    # Update parameters
    optimizer.step()
    optimizer.zero_grad()

    # Update learning rate
    if lr_scheduler is not None:
        lr_scheduler.step()

    return loss


def train(forward_step_func, model, optimizer, lr_scheduler, train_data_iterator, valid_data_iterator):
    """Train the model function with dynamic parallelism."""
    args = get_args()
    timers = get_timers()

    # Initialize dynamic strategy selector if not already done
    if not hasattr(model, 'strategy_selector'):
        from megatron.core.strategy_selector import DynamicStrategySelector
        model.strategy_selector = DynamicStrategySelector(model)

    # Turn on training mode
    for m in model:
        m.train()

    # Tracking variables
    total_loss_dict = {}
    iteration = args.iteration

    timers('interval-time').start()
    while iteration < args.train_iters:
        loss = train_step(forward_step_func, train_data_iterator, model, optimizer, lr_scheduler)
        iteration += 1

        # Logging
        if iteration % args.log_interval == 0:
            elapsed_time = timers('interval-time').elapsed()
            elapsed_time_per_iteration = elapsed_time / args.log_interval
            if args.log_throughput:
                throughput = args.global_batch_size / elapsed_time_per_iteration
                print(f'Iteration {iteration} | Throughput: {throughput:.2f} samples/sec')

            # Log current parallel strategy
            if hasattr(model, 'strategy_selector'):
                strategy = model.strategy_selector.current_strategy
                print(f'Current parallel strategy: TP={strategy["tp_size"]}, '
                      f'PP={strategy["pp_size"]}, DP={strategy["dp_size"]}')


        # Checkpointing
        if args.save and args.save_interval and \
           iteration % args.save_interval == 0:
            save_checkpoint(iteration, model, optimizer, lr_scheduler)

        # Evaluation
        if args.eval_interval and iteration % args.eval_interval == 0:
            prefix = 'iteration {}'.format(iteration)
            evaluate_and_print_results(prefix, forward_step_func,
                                     valid_data_iterator, model,
                                     iteration, False)

    return iteration


def train_valid_test_datasets_provider(train_val_test_num_samples):
    """Build train, valid, and test datasets."""
    args = get_args()
    train_ds, valid_ds, test_ds = None, None, None

    print_rank_0('> building train, validation, and test datasets for GPT ...')
    # Option 1 of data loading using --data-path

    if args.data_path:
        train_ds, valid_ds, test_ds = build_train_valid_test_datasets(
            data_prefix=args.data_path,
            data_impl=args.data_impl,
            splits_string=args.split,
            train_valid_test_num_samples=train_val_test_num_samples,
            seq_length=args.seq_length,
            seed=args.seed,
            skip_warmup=(not args.mmap_warmup))
    # Option 2 of data loading using --(train|valid|test)-weighted-split-paths
    elif args.train_weighted_split_paths:
        assigned_train_valid_test = []
        if args.train_weighted_split_paths is not None:
            train_ds = []
            assigned_train_valid_test.append("train")
        if args.valid_weighted_split_paths is not None:
            valid_ds = []
            assigned_train_valid_test.append("valid")
        if args.test_weighted_split_paths is not None:
            test_ds = []
            assigned_train_valid_test.append("test")

        for s in assigned_train_valid_test:
            data_groups = zip(eval(f"args.{s}_weighted_split_paths"),
                                eval(f"args.{s}_weighted_split_weights"),
                                eval(f"args.{s}_weighted_split_splits"),
                                eval(f"args.{s}_weighted_split_names"))
            for paths, weights, splits, name in data_groups:
                d = build_dataset_group(name, paths, weights, splits,
                                        args.data_impl,
                                        train_val_test_num_samples,
                                        args.seq_length, args.seed,
                                        (not args.mmap_warmup),
                                        train_valid_test=s)
                eval(f"{s}_ds").append(d)
    else:
        raise NotImplementedError("No dataloading argument passed")

    print_rank_0("> finished creating GPT datasets ...")
    return train_ds, valid_ds, test_ds

@record
def main():
    pretrain(train_valid_test_datasets_provider, model_provider, forward_step,
             args_defaults={'tokenizer_type': 'GPT2BPETokenizer'})

if __name__ == "__main__":
    main()
