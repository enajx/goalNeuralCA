import os
import yaml
import wandb
import petname
import torch.distributed as dist
import torch
import pathlib
import socket
import shutil

from src.utils.utils import seed_python_numpy_torch_cuda

if __name__ == "__main__":

    # Load configuration file
    import argparse

    parser = argparse.ArgumentParser(description="Configuration file path")
    parser.add_argument(
        "--conf",
        type=str,
        # default="config_growing.yml",
        # default="config_morphing.yml",
        default="config_translation.yml",
        # default="config_mNCA.yml",
        # default="config_faces.yml",
        metavar="",
        help="Path to yaml configuration file",
    )

    parser.add_argument(
        "--backend",
        type=str,
        default="nccl",
        metavar="",
        help="Parallelization backend",
    )
    parser.add_argument(
        "--init",
        type=str,
        default="env://",
        metavar="",
        help="Parallelization init",
    )
    args = parser.parse_args()

    with open(args.conf) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    dataset = config["task"]

    # Calculate the seed
    seed = seed_python_numpy_torch_cuda(config["seed"])
    config["seed"] = seed

    # Save hostname
    hostname = socket.gethostname()
    config["hostname"] = hostname

    # Check if we're running in distributed mode
    if "LOCAL_RANK" in os.environ:
        distributed = True
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        rank = int(os.environ["RANK"])
    else:
        distributed = False
        local_rank = 0
        world_size = 1
        rank = 0

    if torch.cuda.is_available():
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(local_rank)
    else:
        device = "cpu"

    if distributed:
        dist.init_process_group(backend=args.backend, init_method=args.init)
        print(f"Running in distributed mode on rank {rank}.")
    else:
        print("Running in single GPU mode.")

    if rank == 0:
        if torch.cuda.is_available():
            print("Available CUDA devices:")
            for i in range(torch.cuda.device_count()):
                print(f"\nDevice available:")
                print(f"Device ID: {i}, Device Name: {torch.cuda.get_device_name(i)}")
        else:
            print("No CUDA devices available.")

    if rank == 0:
        if config["checkpoint"]:
            print(f"Resuming training from checkpoint {config['checkpoint']}")
            with open(config["checkpoint"] + "/_config.yml", "r") as file:
                config_checkpoint = yaml.load(file, Loader=yaml.FullLoader)
            id_ = config_checkpoint.get("id", wandb.util.generate_id())
            run_name = config_checkpoint.get("run_name", config.get("run_name") or petname.Generate(3))
        else:
            config_checkpoint = {}
            id_ = wandb.util.generate_id()
            run_name = config.get("run_name") or petname.Generate(3)
    else:
        # Initialize variables so that all ranks have them defined before broadcast
        id_ = None
        run_name = None

    # Broadcast the id_ from rank 0 to all other ranks
    if distributed:
        id_list = [id_]  # Wrap in a list to use broadcast_object_list
        dist.broadcast_object_list(id_list, src=0)
        id_ = id_list[0]

        # If you also need to broadcast run_name:
        run_name_list = [run_name]
        dist.broadcast_object_list(run_name_list, src=0)
        run_name = run_name_list[0]

    if rank == 0:
        # Create new run on WandB  # https://docs.wandb.ai/ref/python/init
        mode = config["wandb_mode"]

        # Create a copy of config without certain keys for wandb
        config_for_wandb = {k: v for k, v in config.items() if k not in ["_path", "id", "seed"]}

        wandb.init(
            project=config["project_name"],
            config=config_for_wandb,
            reinit=True,
            allow_val_change=True,
            mode=mode,
            name=run_name,
            entity=config["entity"],
            resume="allow",
            id=id_,
        )
        wandb.save(args.conf)
        # Keep the local config separate from wandb.config to avoid adding filtered keys back
        # config = wandb.config  # Removed this line

    else:
        wandb.config = config  # Ensure other ranks have access to config

    # Update config with id and path on all ranks (local config only)
    config["id"] = id_
    if config["checkpoint"]:
        config["_path"] = config["checkpoint"]
    else:
        print(f"\nModel ID: {config['id']} and run name: {run_name}\n")
        path = f"{config['saving_path']}/{dataset}/{run_name}"
        config["_path"] = path
        pathlib.Path(path).mkdir(parents=True, exist_ok=True)

    if rank == 0:
        print(f"\nConfig:\n{config}")
        with open(path + f"/_config.yml", "w") as file:
            yaml.dump(dict(config), file)
        # Save original config file with exact formatting
        shutil.copy2(args.conf, path + f"/config_og.yml")
        # Save original config file to wandb
        wandb.save(path + f"/config_og.yml")

    if (
        config["augmentation"]
        and config["noise_task_dim"] == 0
        and config["emb_model_image"] is None
        and config["stochastic_update_ratio"] == 1
        and config["seed_mode"] != "random"
    ):
        raise ValueError("noise_task_dim must be > 0 if augmentation is True")

    if config["augmentation"] and dataset == "faces":
        raise ValueError("augmentation not supported for faces dataset")

    from src.trainers.trainer import train as train_patterns

    train_patterns(
        config,
        rank=rank,
        local_rank=local_rank,
        device=device,
        distributed=distributed,
    )

    # Destroy the process group if distributed
    if distributed:
        dist.destroy_process_group()

    if rank == 0:
        print(f"\nData saved in {config['saving_path']} with run name: {run_name} and id {id_}\n")

    print(config)


# Run with MultiGPU:
# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1,2 torchrun --nproc_per_node=2 train.py --conf=config_patterns.yml

# Run with SingleGPU:
# OMP_NUM_THREADS=4 CUDA_VISIBLE_DEVICES=1 python train.py --conf=config_patterns.yml
