import os
from accelerate.utils import set_seed

def early_setup():
    from config import get_args
    args = get_args()

    # Set which GPUs to be visible to this process
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # print(f"CUDA_VISIBLE_DEVICES set to: {os.getenv('CUDA_VISIBLE_DEVICES')}")
    set_seed(args.seed)
    return args

args_sys = early_setup()

from torch.optim.lr_scheduler import MultiStepLR
import numpy as np
import torch
from torch.nn.utils import clip_grad_norm_
import torch.distributed as dist
import tqdm
from torch.multiprocessing import spawn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from dataloader import MyDataset, generate_data
from framework import MyModel
from tools import custom_collate, setup_ddp, save_checkpoint, get_config, get_time_str, load_checkpoint, get_mapper, \
    calculate_acc, calculate_mrr, find_free_port


def run_stage2(rank, world_size, args, port):
    dataset_path = args.dataset
    config_path = os.path.join(dataset_path, 'settings.yml')
    config = get_config(config_path, easy=True)
    device = 'cuda:' + str(rank)

    if world_size > 0:
        setup_ddp(rank, world_size, port)
    model = MyModel(args, device=device, config=config, mode='stage2_train')
    dataset = MyDataset(args=args, load_mode='stage2_train')
    # print(model)

    if rank == 0: 
        total_params = 0
        total_trainable_params = 0
        lora_trainable_params = 0

        for name, param in model.named_parameters():
            num_params = param.numel()
            total_params += num_params

            if param.requires_grad:
                total_trainable_params += num_params  
                if 'lora' in name:
                    lora_trainable_params += num_params
                else:
                    print(f"{name} requires gradient optimization, Parameters: {num_params}")

        print(f"Total parameters (including frozen): {total_params:,}")
        print(f"Trainable parameters (including LoRA): {total_trainable_params:,}")

        if args.lora:
            model.llm.model.print_trainable_parameters()

    if world_size > 0:
        model = DistributedDataParallel(model, device_ids=[rank], static_graph=True)

    # Optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-5)
    scheduler = MultiStepLR(
        optimizer,
        milestones=[6, 9, 12, 15, 18],
        gamma=0.5
    )
    
    loss_fn = torch.nn.NLLLoss()

    if world_size > 0:
        sampler = DistributedSampler(dataset, shuffle=True)
        dataloader = DataLoader(dataset, batch_size=args.batch, sampler=sampler, drop_last=False,
                                collate_fn=lambda x: custom_collate(x))
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch, drop_last=False, shuffle=True,
                                collate_fn=lambda x: custom_collate(x))

    for epoch in range(args.epoch):
        epoch_loss = 0
        model.train()

        if world_size > 0:
            sampler.set_epoch(epoch)

        if args.verbose:
            disable = rank != 0
        else:
            disable = True

        for batch in tqdm.tqdm(dataloader, desc=f'Training [{epoch+1}/{args.epoch}]', disable=disable, mininterval=500):
            loc_label = batch['loc_label']
            loc_label = torch.tensor(loc_label, device=device)
            if args.aux_loss_expert:
                loc_out, lb_loss = model(batch)
            else:
                loc_out = model(batch)
            loc_loss = loss_fn(loc_out, loc_label)
            total_loss = loc_loss.sum()
            if args.aux_loss_expert:
                total_loss += 10*lb_loss.sum()

            optimizer.zero_grad()
            total_loss.backward()
            # clip_grad_norm_(model.parameters(), 1)
            optimizer.step()
            epoch_loss += total_loss.item()

        if rank == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f'========{get_time_str()}========')
            print(
                f"Epoch {epoch + 1}/{args.epoch}, "
                f"Samples: {len(dataset)}, "
                f"Loss: {round(epoch_loss / len(dataloader), 4)}, Learning Rate: {current_lr}")
        # if rank == 0:
        #     save_checkpoint(model.module if world_size > 0 else model, args, epoch=epoch+1)
        scheduler.step()
        if (epoch+1) % args.test_epoch == 0 and epoch+1 > 0:
            mrr_loc = inference(rank, world_size, args, port, model)


def inference(rank, world_size, args, port, model=None):
    device = 'cuda:' + str(rank)
    dataset_path = args.dataset
    config_path = os.path.join(dataset_path, 'settings.yml')
    config = get_config(config_path, easy=True)
    dataset = MyDataset(args=args, load_mode='stage2_test')

    if model is None:
        if world_size > 0:
            setup_ddp(rank, world_size, port)
        model = MyModel(args, device=device, config=config, mode='stage2_test')
        load_checkpoint(model, args, device=device)
        if world_size > 0:
            model = DistributedDataParallel(model, device_ids=[rank], static_graph=True)
    if world_size > 0:
        sampler = DistributedSampler(dataset, shuffle=False)
        dataloader = DataLoader(dataset, batch_size=args.batch,
                                sampler=sampler,
                                drop_last=False,
                                collate_fn=lambda x: custom_collate(x))
    else:
        dataloader = DataLoader(dataset, batch_size=args.batch,
                                shuffle=False,
                                drop_last=False,
                                collate_fn=lambda x: custom_collate(x))
    if rank == 0:
        print(f"Test Total samples: {len(dataset)}")

    model.eval()
    total = 0
    top_k_values = [1, 2, 3, 5, 10, 20]
    top_k_correct_loc = np.array([0 for _ in range(len(top_k_values))])
    precision_loc = 0
    with torch.no_grad():
        if args.verbose:
            disable = rank != 0
        else:
            disable = True
        for batch in tqdm.tqdm(dataloader, disable=disable, mininterval=500):
            loc_labels = batch['loc_label']
            loc_labels = torch.tensor(loc_labels, device=device)

            if args.aux_loss_expert:
                loc_out, _ = model(batch)
            else:
                loc_out = model(batch)
            top_k_correct_loc += calculate_acc(loc_out, loc_labels, top_k_values)
            precision_loc += calculate_mrr(loc_out, loc_labels)
            total += len(loc_labels)

    total_samples_tensor = torch.tensor(total, device=device)
    top_k_correct_loc_tensor = torch.tensor(top_k_correct_loc, device=device)
    precision_loc_tensor = torch.tensor(precision_loc, device=device)

    # Distributed aggregation using all_reduce
    if world_size > 0:
        dist.all_reduce(total_samples_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(top_k_correct_loc_tensor, op=dist.ReduceOp.SUM)
        dist.all_reduce(precision_loc_tensor, op=dist.ReduceOp.SUM)

    # Convert aggregated tensors back to local variables
    total_samples = total_samples_tensor.item()
    top_k_correct_loc = top_k_correct_loc_tensor.cpu().numpy()
    precision_loc = precision_loc_tensor.item()

    # Calculate final metrics
    top_k_accuracy_loc = top_k_correct_loc / total_samples * 100
    mrr_loc = precision_loc * 100 / total_samples

    # Only rank 0 prints or saves results
    if rank == 0:
        print(f"Test Loc Results - Total Samples: {total_samples}")
        for k, accuracy in zip(top_k_values, top_k_accuracy_loc):
            print(f"Acc@{k}: {accuracy:.2f}")
        print(f"MRR: {mrr_loc:.2f}\n")
    return mrr_loc
    # else:
    #     return None

def main(args_main):
    get_mapper(args_main.dataset)
    if args_main.gpu:
        n_gpus = len(args_main.gpu.split(','))
    else:
        n_gpus = 1

    master_port = find_free_port()
    print(f"[main] Picked master_port={master_port}")
    generate_data(args_main, 'stage2_train')
    generate_data(args_main, 'stage2_test')

    print(f'***** Using LLM: {args_main.llm_id} *****')
    print(f'***** Using Dataset: {args_main.dataset} *****')

    if args_main.inference:
        if n_gpus > 1:
            spawn(inference, args=(n_gpus, args_main, master_port), nprocs=n_gpus)
        else:
            inference(rank=0, world_size=0, args=args_main, port=master_port)
        exit()

    if n_gpus > 1:
        spawn(run_stage2, args=(n_gpus, args_main, master_port), nprocs=n_gpus)
    else:
        run_stage2(rank=0, world_size=0, args=args_main, port=master_port)

if __name__ == "__main__":
    main(args_sys)
