---
author: Alexandre Strube // Sabrina Benassou 
title: Bringing Deep Learning Workloads to JSC supercomputers
subtitle: Parallelize Training
date: June 25th, 2025
---

## Before Starting

- Move to the correct folder
    
    ```bash
    cd 2025-06-course-Bringing-Deep-Learning-Workloads-to-JSC-supercomputers/code/parallelize/
    ```

---

## What this code does 

- It trains a [transformer](https://arxiv.org/pdf/1706.03762) model on the [xsum](https://paperswithcode.com/dataset/xsum) dataset to summarize documents.
- **Transformers** is a deep learning model architecture that uses self-attention to process sequences in parallel.
- **XSum** is a dataset for abstractive text summarization, containing news articles and their summaries.

---

## What this code does 

- Again, this is not a deep learning course.
- If you are not familiar with the model and the dataset, just imagine it as a black box: you provide it with text, and it returns a summary.

    ![](images/black_box.svg)

---

## Libraries

- You already downloaded the libraries yesterday that we will use today:
    - **PyTorch:** A deep learning framework for building and training models.
    - **Hugging Face:** A platform and library for natural language processing (NLP) models and datasets.
    - **Transformers:** A library by Hugging Face for state-of-the-art NLP models.

---

Let's have a look at the files **```train.py```** and **```run_train.sbatch```** in the repo.

![](images/look.jpg)

---

## Run the Training Script

- There are TODOs in these two files. **Do not modify the TODOs for now**. The code is already working, so you don’t need to make any changes at this point.
- Now run:

    ```bash
    sbatch run_train.sbatch 
    ```

- Spoiler alert 🚨

- The code won't work.

- Check the output and error files

---

## What is the problem?

- Remember, there is no internet on the compute node.
- Therefore, you should:
    - **Comment out** lines 90 **to** 140.
    - Activate your environment:

        ```bash
        source $HOME/course/sc_venv_template/activate.sh
        ```

    - Run:

        ```bash
        python train.py
        ```

    - **Uncomment back** lines 90-140.
    - Finally, run your job again 🚀:

        ```bash
        sbatch run_train.sbatch
        ```

---

## JOB Running

- Congrats, you are training a DL model on the supercomputer using one GPU 🎉

--- 

## llview

- You can monitor your training using [llview](https://go.fzj.de/llview-jureca). 
- Use your Judoor credentials to connect.
- Check the job number that you are intrested in.
    ![](images/llview_job.png){height=400px}

---


## llview

- Go to the right to open the PDF document. **It may take some time to load the job information, so please wait until the icon turns blue**.
    ![](images/llview.png){height=450px}

---

## llview

- You have many information about your job once you open the PDF file.
    ![](images/llview_info.png)

---

## GPU utilization 

- You can see that in fact we are using **1 GPU**

    ![](images/llview_gpu_1.png)

---

## GPU utilization   

- It is a waste of resources.

- The training takes time (13m according to llview).

- Then, can we run our model on multiple GPUs ?

---

## What if

- In file **```run_train.sbatch```**, we increase the number of GPUs at line 3 to 4:

    ```bash
    #SBATCH --gres=gpu:4
    ```

- And run our job again

    ```bash
    sbatch run_train.sbatch
    ```

--- 


## llview

- We are still using **1 GPU**

- ![](images/llview_gpu_2.png)

---

## We need communication

- Without correct setup, the GPUs might not be utilized.

- Furthermore, we don't have an established communication between the GPUs

    ![](images/dist/no_comm.svg){height=400px}

---

## We need communication

![](images/dist/comm1.svg){height=500px}

---

## We need communication

![](images/dist/comm2.svg){height=500px}

---

## collective operations

- The GPUs use collective operations to communicate and share data in parallel computing
- The most common collective operations are: All Reduce, All Gather, and Reduce Scatter

---

## All Reduce 

![](images/dist/all_reduce.svg)

- Other operations, such as **min**, **max**, and **avg**, can also be performed using All-Reduce.

---

## All Gather

![](images/dist/all_gather.svg)

--- 

## Reduce Scatter

![](images/dist/reduce_scatter.svg)

--- 

## Terminologies

- Before going further, we need to learn some terminologies

---

## World Size

![](images/dist/gpus.svg){height=550px}

---

## Rank

![](images/dist/rank.svg){height=550px}

---

## local_rank

![](images/dist/local_rank.svg){height=550px}

---

## Now

That we have understood how the devices communicate and the terminologies used in parallel computing, 
we can move on to distributed training (training on multiple GPUs).

---

## Distributed Training

- Parallelize the training across multiple nodes, 
- Significantly enhancing training speed and model accuracy.
- It is particularly beneficial for large models and computationally intensive tasks, such as deep learning.[[1]](https://pytorch.org/tutorials/distributed/home.html)


---

## Distributed Data Parallel (DDP)

[DDP](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html) is a method in parallel computing used to train deep learning models across multiple GPUs or nodes efficiently.

![](images/ddp/ddp-2.svg){height=400px}

--- 

## DDP

![](images/ddp/ddp-3.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-4.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-5.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-6.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-7.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-8.svg){height=500px}

--- 

## DDP

![](images/ddp/ddp-9.svg){height=500px}

--- 

## DDP

If you're scaling DDP to use multiple nodes, the underlying principle remains the same as single-node multi-GPU training.

---

## DDP

![](images/ddp/multi_node.svg){height=500px}

---

## DDP recap

- Each GPU on each node gets its own process.
- Each GPU has a copy of the model.
- Each GPU has visibility into a subset of the overall dataset and will only see that subset.
- Each process performs a full forward and backward pass in parallel and calculates its gradients.
- The gradients are synchronized and averaged across all processes.
- Each process updates its optimizer.

---

## Let's start coding!

- Whenever you see **TODOs**💻📝, follow the instructions to either copy-paste the code at the specified line numbers or type it yourself.

- Depending on how you copy and paste, the line numbers may vary, but always refer to the TODO numbers in the code and slides.

---

## Setup communication

- We need to setup a communication among the GPUs. 
- For that we would need the file **```distributed_utils.py```**.
- **TODOs**💻📝:
    1. Import **```distributed_utils```** file at line 13:
        
        ```python 
        # This file contains utility_functions for distributed training.
        from distributed_utils import *
        ```
    2. Then **remove** lines 77 and 78:

        ```python
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        ```
    3. and **add** at line 77 a call to the method **```setup()```** defined in **```distributed_utils.py```**: 

        ```python
        # Initialize a communication group and return the right identifiers.
        local_rank, rank, device = setup()
        ```

---

## Setup communication

What is in the **```setup()```** method ?

```python
def setup():
    # Initializes a communication group using 'nccl' as the backend for GPU communication.
    torch.distributed.init_process_group(backend='nccl')
    # Get the identifier of each process within a node
    local_rank = int(os.getenv('LOCAL_RANK'))
    # Get the global identifier of each process within the distributed system
    rank = int(os.environ['RANK'])
    # Creates a torch.device object that represents the GPU to be used by this process.
    device = torch.device('cuda', local_rank)
    # Sets the default CUDA device for the current process, 
    # ensuring all subsequent CUDA operations are performed on the specified GPU device.
    torch.cuda.set_device(device)
    # Different random seed for each process.
    torch.random.manual_seed(1000 + torch.distributed.get_rank())

    return local_rank, rank, device
```

---

## Model

- **TODO 4**💻📝:

    - At line 83, wrap the model in a **DistributedDataParallel** (DDP) module to parallelize the training across multiple GPUs.
    
        ```python 
        # Wrap the model in DistributedDataParallel module 
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[local_rank],
        )
        ```

---

## DistributedSampler 

- **TODO 5**💻📝:

    - At line 94, instantiate a **DistributedSampler** object for each set to ensure that each process gets a different subset of the data.
    
        ```python
        # DistributedSampler object for each set to ensure that each process gets a different subset of the data.
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, 
                                                                        shuffle=True, 
                                                                        seed=args.seed)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_dataset)
        test_sampler = torch.utils.data.distributed.DistributedSampler(test_dataset)
        ```

---

## DataLoader

- **TODO 6**💻📝:

    - At line 103, **REMOVE** **```shuffle=True```** in the DataLoader of train_loader and **REPLACE** it by **```sampler=train_sampler```**
        
        ```python 
        train_loader = DataLoader(train_dataset, 
                                batch_size=args.batch_size, 
                                sampler=train_sampler, # pass the sampler argument to the DataLoader
                                num_workers=int(os.getenv('SLURM_CPUS_PER_TASK')),
                                pin_memory=True)
        ```

---

## DataLoader

- **TODO 7**💻📝:

    -  At line 108, pass **val_sampler** to the sampler argument of the val_dataLoader

        ```python
        val_loader = DataLoader(val_dataset,
                                batch_size=args.batch_size,
                                sampler=val_sampler, # pass the sampler argument to the DataLoader
                                pin_memory=True)
        ```

- **TODO 8**💻📝:

    - At line 112, pass **test_sampler** to the sampler argument of the test_dataLoader

        ```python
        test_loader = DataLoader(test_dataset,
                                batch_size=args.test_batch_size,
                                sampler=test_sampler, # pass the sampler argument to the DataLoader
                                pin_memory=True)    
        ```

--- 

## Sampler 

- **TODO 9**💻📝:

    - At line 125, **set** the current epoch for the dataset sampler to ensure proper data shuffling in each epoch

        ```python
        # Pass the current epoch to the sampler to ensure proper data shuffling in each epoch
        train_sampler.set_epoch(epoch)
        ```

---

## All Reduce Operation

- **TODO 10**💻📝:

    - At **lines 49 and 72**, Obtain the global average loss across the GPUs.

        ```python
        # Return the global average loss.
        torch.distributed.all_reduce(result, torch.distributed.ReduceOp.AVG)
        ```

---

## print

- **TODO 11**💻📝:

    - **Replace** all the ```print``` methods by **```print0```** method defined in **```distributed_utils.py```** to allow only rank 0 to print in the output file.
    
    - At **line 133** 

        ```python
        # We use the utility function print0 to print messages only from rank 0.
        print0(f'[{epoch+1}/{args.epochs}] Train loss: {train_loss:.5f}, validation loss: {val_loss:.5f}')
        ```
    - At **line 144**

        ```python
        # We use the utility function print0 to print messages only from rank 0.
        print0('Finished training after', end_time - start_time, 'seconds.')
        ```
    - At **line 148**
    
        ```python
        # We use the utility function print0 to print messages only from rank 0.
        print0('Final test loss:', test_loss.item())
        ```

---

## print

The definition of the function **print0** is in **```distributed_utils.py```**

```python
functools.lru_cache(maxsize=None)
def is_root_process():
    """Return whether this process is the root process."""
    return torch.distributed.get_rank() == 0


def print0(*args, **kwargs):
    """Print something only on the root process."""
    if is_root_process():
        print(*args, **kwargs)
```

---

## Save model 

- **TODO 12**💻📝:

    - At **lines 138 and 151**, replace torch.save method with the utility function save0 to allow only the process with rank 0 to save the model.
 
        ```python 
        # We allow only rank=0 to save the model
        save0(model, 'model-best')
        ```
        ```python 
        # We allow only rank=0 to save the model
        save0(model, 'model-final')
        ```

---

## Save model 

The method **save0** is defined in **```distributed_utils.py```**

```python
functools.lru_cache(maxsize=None)
def is_root_process():
    """Return whether this process is the root process."""
    return torch.distributed.get_rank() == 0


def save0(*args, **kwargs):
    """Pass the given arguments to `torch.save`, but only on the root
    process.
    """
    # We do *not* want to write to the same location with multiple
    # processes at the same time.
    if is_root_process():
        torch.save(*args, **kwargs)
```

--- 

## We are almost there

- That's it for the **train.py** file. 
- But before launching our job, we need to add some lines to **run_train.sbatch** file 

---

## Setup communication

In **```run_train.sbatch```** file:

- **TODOs 13**💻📝: 
    - At line 3, increase the number of GPUs to 4 if it is not already done.

        ```bash
        #SBATCH --gres=gpu:4
        ```

    - At line 22, pass the correct number of devices.

        ```bash
        export CUDA_VISIBLE_DEVICES=0,1,2,3
        ```

---

## Setup communication

Stay in **```run_train.sbatch```** file:

- **TODO 14**💻📝: we need to setup **MASTER_ADDR** and **MASTER_PORT** to allow communication over the system.

    - At line 24, add the following:

        ```bash
        # Extracts the first hostname from the list of allocated nodes to use as the master address.
        MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)"
        # Modifies the master address to allow communication over InfiniBand cells.
        MASTER_ADDR="${MASTER_ADDR}i"
        # Get IP for hostname.
        export MASTER_ADDR="$(nslookup "$MASTER_ADDR" | grep -oP '(?<=Address: ).*')"
        export MASTER_PORT=7010
        ```

---

## Setup communication

We are not done yet with **```run_train.sbatch```** file:

- **TODO 15**💻📝: 
    
    - At line 35, we change the lauching script to use **torchrun_jsc** and pass the following argument: 

        ```bash
        # Launch a distributed training job across multiple nodes and GPUs
        srun --cpu_bind=none bash -c "torchrun_jsc \
            --nnodes=$SLURM_NNODES \
            --rdzv_backend c10d \
            --nproc_per_node=gpu \
            --rdzv_id $RANDOM \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            --rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi) \
            train.py "
        ```

---

## Setup communication

- The arguments that we pass are:

    1. **```nnodes=$SLURM_NNODES```**: the number of nodes
    2. **```rdzv_backend c10d```**: the c10d method for coordinating the setup of communication among distributed processes.
    3. **```nproc_per_node=gpu```** the number of GPUs
    4. **```rdzv_id $RANDOM```** a random id which that acts as a central point for initializing and coordinating the communication among different nodes participating in the distributed training. 
    5. **```rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT```** the IP that we setup in the previous slide to ensure all nodes know where to connect to start the training session.
    6. **```rdzv_conf=is_host=\$(if ((SLURM_NODEID)); then echo 0; else echo 1; fi)```** The rendezvous host which is responsible for coordinating the initial setup of communication among the nodes.

---

## done ✅

- You can finally run:

    ```bash
    sbatch run_train.sbatch
    ```

---

## llview

- Let's have a look at our job using [llview](https://go.fzj.de/llview-jureca) again.

- You can see that now, we are using all the GPUs of the node

- ![](images/llview_gpu_4.png)

--- 

## llview

- And that our job took less time to finish training (4m vs 13m with one GPU)

- And even the test loss function is lower (0.538 vs 0.636 with one GPU). 

- But what about using more nodes ?

---

## What about using more nodes ?

---

## Multi-node training

- In **```run_train.sbatch```** at line 2, you can increase the number of nodes to 2:

    ```bash
    #SBATCH --nodes=2
    ```

- Hence, you will use 8 GPUs for training.

- Run again:

    ```bash
    sbatch run_train.sbatch
    ```

--- 

## llview

- Open [llview](https://go.fzj.de/llview-jureca) again.

- You can see that now, we are using 2 nodes and 8 GPUs.

- ![](images/llview_gpu_8.png)

---

## Amazing ✨

---

## Before we go further...

- Data parallel is usually good enough 👌 
- However, if your model is too big to fit into a single GPU
- Welllll ... there other distributed techniques ...

---

## Fully Sharded Data Parallel (FSDP)

![](images/fsdp/fsdp-0.svg){height=375pt}

---

## FSDP

![](images/fsdp/fsdp-1.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-2.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-3.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-4.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-5.svg){height=425pt}

---

## FSDP

![](images/fsdp/fsdp-5-5.svg){height=425pt}

---

## FSDP

![](images/fsdp/fsdp-6.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-7.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-8.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-9.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-10.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-11.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-12.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-13.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-14.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-15.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-16.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-17.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-18.svg){height=425pt}

---

## FSDP

![](images/fsdp/fsdp-19.svg){height=425pt}

---

## FSDP

![](images/fsdp/fsdp-20.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-21.svg){height=425pt}



---

## FSDP

![](images/fsdp/fsdp-22.svg){height=425pt}



---

## FSDP

![](images/fsdp/fsdp-23.svg){height=425pt}



---

## FSDP

![](images/fsdp/fsdp-24.svg){height=425pt}



---

## FSDP

![](images/fsdp/fsdp-25.svg){height=425pt}



---

## FSDP

![](images/fsdp/fsdp-26.svg){height=425pt}



---

## FSDP

![](images/fsdp/fsdp-27.svg){height=425pt}


---

## FSDP

![](images/fsdp/fsdp-28.svg){height=425pt}

---

## FSDP

- FSDP is a primitive method in PyTorch.
- Its memory efficiency is high because model parameters, gradients and optimizers are sharded.
- However, it requires a high-bandwidth system because it involves frequent communication between GPUs.
- If you have bandwidth-limited clusters, FSDP may not be ideal, and you would prefer pipelining technique.

---

## Model Parallel

- Before talking about pipelining, let's talk about Model Parallelism (MP).
- Model *itself* is too big to fit in one single GPU 🐋
- Each GPU holds a slice of the model 🍕
- Data moves from one GPU to the next

---

## Model Parallel

![](images/model-parallel.svg)

---


## Model Parallel

![](images/model-parallel-pipeline-1.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-2.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-3.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-4.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-5.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-6.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-7.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-8.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-9.svg)

---

## Model Parallel

![](images/model-parallel-pipeline-10.svg)

---

## What's the problem here? 🧐

---

## Model Parallel

- Waste of resources
- While one GPU is working, others are waiting the whole process to end
- ![](images/no_pipe.png)
    - [Source: GPipe: Efficient Training of Giant Neural Networks using Pipeline Parallelism](https://arxiv.org/abs/1811.06965)


---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-1.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-2-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-3-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-4-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-5-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-6-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-7-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-8-multibatch.svg)

---

## Model Parallel - Pipelining

![](images/model-parallel-pipeline-9-multibatch.svg)

---

## This is an oversimplification!

- Actually, you split the input minibatch into multiple microbatches.
- There's still idle time - an unavoidable "bubble" 🫧
- ![](images/pipe.png)

---

## Model Parallel - Multi Node

- In this case, each node does the same as the others. 
- At each step, they all synchronize their weights.

---

## Model Parallel - Multi Node

![](images/model-parallel-multi-node.svg)

---

## Pipeline Parallelism

- Pipeline parallelism does not require frequent communication because the model is stored sequentially in stages.
- If your model is computationally intensive with extremely wide layers, you may consider Tensor Parallelism (TP).

---

## Tensor Parallelism (TP)

![](images/tp/tp-1.png)

---

## TP

![](images/tp/tp-2.png)

---

## TP

![](images/tp/tp-3.png)


---

## TP

![](images/tp/tp-4.png)


---

## TP

![](images/tp/tp-5.png)

---

## TP 

- We have introduced row parallelism.
- There is also column parallelism, where the weight columns are split across GPUs.
- Tensor Parallelism (TP) is great for large, compute-heavy layers like matrix multiplications.
- However, TP requires frequent communication during tensor operations.

---

## 3D Parallelism

![[3D Parallelism](https://arxiv.org/pdf/2410.06511)](images/3dp.png)

- 3D Parallelism combines Tensor Parallelism (TP), Pipeline Parallelism (PP), and Data Parallelism (DP) to efficiently train large models by distributing computation, memory, and data across multiple GPUs. 
- It enables scaling to very large models by addressing compute, memory, and communication bottlenecks in a balanced way.

---

## Day 2 RECAP 

- You know where to store your code and your data. 🗂️
- How to create HDF5 and PyArrow files. 📄
- You know what distributed training is. 🧑‍💻
- You can submit training jobs on a single GPU, multiple GPUs, or across multiple nodes. 🎮💻
- You are familiar with DDP and aware of other distributed training techniques like FSDP, TP, PP, and 3D parallelism. ⚙️💡
- You know how to monitor your training using llview. 📊👀

---

## Find Out More

- Here are some useful:

    - Papers:
        - [FSDP paper](https://arxiv.org/pdf/2304.11277)
        - [Pipeline Parallelism](https://arxiv.org/pdf/1811.06965)
        - [Tensor Parallelism](https://arxiv.org/pdf/1909.08053)

    - Tutorials:
        - [PyTorch at JSC](https://sdlaml.pages.jsc.fz-juelich.de/ai/recipes/pytorch_at_jsc/)
        - [PyTorch tutorials GitHub](https://github.com/pytorch/tutorials/tree/main)
        - [PyTorch documentation](https://pytorch.org/tutorials/distributed/home.html)

    - Links
        - [AI Landing Page](https://sdlaml.pages.jsc.fz-juelich.de/ai/)
        - [Other courses at JSC](https://www.fz-juelich.de/en/ias/jsc/education/training-courses)


---

## ANY QUESTIONS??

#### Feedback is more than welcome!

---

