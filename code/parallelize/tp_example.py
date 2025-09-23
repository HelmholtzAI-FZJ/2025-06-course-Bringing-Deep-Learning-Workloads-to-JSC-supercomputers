"""
This is the script to test 2D Parallel which combines Tensor/Sequence
parallel with Fully Sharded Data Parallel (TP/SP + FSDP) on a example
Llama2 model. We show an E2E working flow from forward, backward
and optimization.

We enabled Fully Sharded Data Parallel + Tensor Parallel in
separate parallel dimensions:
    Data Parallel ("dp") across hosts
    Tensor Parallel ("tp") within each host

 We use a simple diagram to illustrate below:

======================================================================
------------       ------------       ------------       ------------
| Host 1   |       | Host 2   |       |          |       | Host N   |
| 8 GPUs   |       | 8 GPUs   |       |          |       | 8 GPUs   |
|          |       |          |       |    ...   |       |          |
| (TP)     |       | (TP)     |       |          |       | (TP)     |
|[0,1,..,7]|       |[8,9..,15]|       |          |       |[8N-8,8N-7|
|          |       |          |       |          |       | .., 8N-1]|
|          |       |          |       |          |       |          |
------------       ------------       ------------       ------------
FSDP:
[0, 8, ..., 8N-8], [1, 9, ..., 8N-7], ..., [7, 15, ..., 8N-1]
======================================================================

More details can be seen in the PyTorch tutorials:
https://pytorch.org/tutorials/intermediate/TP_tutorial.html
"""

import sys
import os
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from log_utils import rank_log, get_logger, verify_min_gpu_count

# ---- GPU check ------------
_min_gpu_count = 4

if not verify_min_gpu_count(min_gpus=_min_gpu_count):
    print(f"Unable to locate sufficient {_min_gpu_count} gpus to run this example. Exiting.")
    sys.exit()
# ---------------------------

from llama2_model import Transformer, ModelArgs

from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import fully_shard
from torch.distributed._tensor import Shard, Replicate
from torch.distributed.tensor.parallel import (
    parallelize_module,
    ColwiseParallel,
    RowwiseParallel,
    PrepareModuleInput,
    SequenceParallel
)

import os
import random
import numpy as np
import torch

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# cuBLAS determinism (must be set before CUDA context is created)
os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":16:8")

# Fix NCCL algorithm/protocol so reduction order doesn't change run-to-run
os.environ.setdefault("NCCL_ALGO", "Tree")
os.environ.setdefault("NCCL_PROTO", "Simple")

# (Optional but useful for debugging / stability)
os.environ.setdefault("NCCL_BLOCKING_WAIT", "1")
os.environ.setdefault("TORCH_SHOW_CPP_STACKTRACES", "1")


# Global, rank-invariant seed for weight init & op RNG
GLOBAL_SEED = 12345

random.seed(GLOBAL_SEED)
np.random.seed(GLOBAL_SEED)
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)

# Disable runtime autotuning that can introduce nondeterminism
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Turn off TF32 so math is identical across runs
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# Hard fail if a non-deterministic op sneaks in
torch.use_deterministic_algorithms(True)


tp_size = 2
logger = get_logger()

# understand world topology
# _rank = int(os.environ["RANK"])
# _world_size = int(os.environ["WORLD_SIZE"])


# print(f"Starting PyTorch 2D (FSDP + TP) example on rank {_rank}.")
# assert (
#     _world_size % tp_size == 0
# ), f"World size {_world_size} needs to be divisible by TP size {tp_size}"


# create a sharding plan based on the given world_size.
# dp_size = _world_size // tp_size

# device = torch.accelerator.current_accelerator().type
# Create a device mesh with 2 dimensions.
# First dim is the data parallel dimension
# Second dim is the tensor parallel dimension.
# device_mesh = init_device_mesh(device, (dp_size, tp_size), mesh_dim_names=("dp", "tp"))

# rank_log(_rank, logger, f"Device Mesh created: {device_mesh=}")
# tp_mesh = device_mesh["tp"]
# dp_mesh = device_mesh["dp"]

# For TP, input needs to be same across all TP ranks.
# while for SP, input can be different across all ranks.
# We will use dp_rank for setting the random seed
# to mimic the behavior of the dataloader.
# dp_rank = dp_mesh.get_local_rank()

# Ensure identical parameter init across runs
torch.manual_seed(GLOBAL_SEED)
torch.cuda.manual_seed_all(GLOBAL_SEED)

simple_llama2_config = ModelArgs(dim=256, n_layers=2, n_heads=16, vocab_size=32000)
model = Transformer.from_model_args(simple_llama2_config).to(device)
model.init_weights()


# # parallelize the first embedding and the last linear out projection
# model = parallelize_module(
#     model,
#     tp_mesh,
#     {
#         "tok_embeddings": RowwiseParallel(
#             input_layouts=Replicate(),
#             output_layouts=Shard(1),
#         ),
#         "norm": SequenceParallel(),
#         "output": ColwiseParallel(
#             input_layouts=Shard(1),
#             output_layouts=Replicate()
#         ),
#         "norm4": SequenceParallel(),
#     }
# )

# for layer_id, transformer_block in enumerate(model.layers):
#     layer_tp_plan = {
#         "attention_norm": SequenceParallel(),
#         "attention": PrepareModuleInput(
#             input_layouts=(Shard(1), Replicate()),
#             desired_input_layouts=(Replicate(), Replicate()),
#         ),
#         "attention.wq": ColwiseParallel(use_local_output=False),
#         "attention.wk": ColwiseParallel(use_local_output=False),
#         "attention.wv": ColwiseParallel(use_local_output=False),
#         "attention.wo": RowwiseParallel(output_layouts=Shard(1)),
#         "ffn_norm": SequenceParallel(),
#         "feed_forward": PrepareModuleInput(
#             input_layouts=(Shard(1),),
#             desired_input_layouts=(Replicate(),),
#         ),
#         "feed_forward.w1": ColwiseParallel(),
#         "feed_forward.w2": RowwiseParallel(output_layouts=Shard(1)),
#         "feed_forward.w3": ColwiseParallel(),
#     }

#     # Custom parallelization plan for the model
#     parallelize_module(
#         module=transformer_block,
#         device_mesh=tp_mesh,
#         parallelize_plan=layer_tp_plan
#     )

# # Init FSDP using the dp device mesh
# sharded_model = fully_shard(model, mesh=dp_mesh)

# rank_log(_rank, logger, f"Model after parallelization {sharded_model=}\n")

# Create an optimizer for the parallelized and sharded model.
lr = 3e-3
# rank_log(_rank, logger, f"Creating AdamW optimizer with learning rate {lr}")
# Before creating the optimizer:
# torch._C._distributed_c10d._set_c10d_bucket_cap_mb(1)  # small buckets => more stable order

optimizer = torch.optim.AdamW(model.parameters(), lr=lr, foreach=False)


# Training loop:
# Perform a num of iterations of forward/backward
# and optimizations for the sharded module.
# rank_log(_rank, logger, "\nStarting 2D training...")
num_iterations = 10
batch_size = 2

# One generator per DP replica, stable across runs
data_gen = torch.Generator(device=device)
data_gen.manual_seed(10_000)  # any fixed mapping from dp_rank -> seed

num_iterations = 10
for i in range(num_iterations):
    # Stable inputs across runs (and stable-but-different per DP replica)
    inp = torch.randint(32000, (8, 256), device=device, generator=data_gen)

    output = model(inp)
    output.sum().backward()
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)  # explicit, consistent

    # rank_log(_rank, logger, f"2D iter {i} complete")
    # if _rank == 0:
    import hashlib
    def tensor_digest(t):
        return hashlib.sha256(t.detach().float().cpu().numpy().tobytes()).hexdigest()[:16]

    # Check a slice of output and a param digest
    print("output[0,0,0:5] =", output[0,0,0:5])
    first_param = next(model.parameters())
        # print("param_digest:", tensor_digest(first_param))


# rank_log(_rank, logger, "2D training successfully completed!")

print("Example complete!")

if dist.is_initialized():
    dist.destroy_process_group()
    