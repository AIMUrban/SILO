import os
import random
import socket
import time
from contextlib import closing

import numpy as np
import pandas as pd
import peft
import torch
import yaml
from easydict import EasyDict
from torch.distributed import init_process_group
from torch.utils.data import Subset
from transformers import AutoTokenizer, AutoModelForCausalLM


def custom_collate(batches):
    batches = {
        'user': [item['user'] for item in batches],
        'loc_his': np.array([item['loc_his'] for item in batches]),
        'loc_cur': np.array([item['loc_cur'] for item in batches]),
        'timeslot_his': np.array([item['timeslot_his'] for item in batches]),
        'timeslot_cur': np.array([item['timeslot_cur'] for item in batches]),
        'loc_label': [item['loc_label'] for item in batches],
        # 'time_str_his': [item['time_str_his'] for item in batches],
        # 'time_str_cur': [item['time_str_cur'] for item in batches],
        # 'weekday_str_his': [item['weekday_str_his'] for item in batches],
        # 'weekday_str_cur': [item['weekday_str_cur'] for item in batches],
        # 'confidence': [item['confidence'] for item in batches],
        'user_json': [item['user_json'] for item in batches],
    }
    return batches

def calculate_acc(output, label, ks):
    topk_correct_counts = [
        torch.sum(
            (torch.topk(output, k=top_k, dim=1)[1] + 0) == label.unsqueeze(1)
        ).item()
        for top_k in ks
    ]
    return np.array(topk_correct_counts)

def calculate_mrr(output, true_labels):
    res = 0.0
    for i, pred in enumerate(output):
        sorted_indices = torch.argsort(pred, descending=True)
        sorted_indices = sorted_indices.cpu()
        true_index = np.where(true_labels[i].cpu() == sorted_indices)[0]
        if len(true_index) > 0:
            res += 1.0 / (true_index[0] + 1)
    return res

def get_mapper(dataset_path):
    location_mapper_path = os.path.join(dataset_path, 'location_mapper.npy')
    user_mapper_path = os.path.join(dataset_path, 'user_mapper.npy')

    if os.path.exists(location_mapper_path):
        return

    train_data = pd.read_parquet(os.path.join(dataset_path, 'train_sampled.parquet'))
    test_data = pd.read_parquet(os.path.join(dataset_path, 'test_sampled.parquet'))
    data = pd.concat([train_data, test_data]).reset_index(drop=True)

    data['loc'] = data.apply(lambda row: f"{row['x']}_{row['y']}", axis=1)

    user_set = data['uid'].unique().tolist()
    location_set = data['loc'].unique().tolist()

    location2id = {location: idx for idx, location in enumerate(location_set)}
    user2id = {user: idx for idx, user in enumerate(user_set)}

    print('unique location num:', len(location2id))
    print('unique user num:', len(user2id))
    update_config(path=os.path.join(dataset_path, 'settings.yml'), key_list=['Dataset', 'num_locations'], value=len(location2id))
    update_config(path=os.path.join(dataset_path, 'settings.yml'), key_list=['Dataset', 'num_users'], value=len(user2id))
    print('update config done')
    np.save(location_mapper_path, location2id)
    np.save(user_mapper_path, user2id)

def update_config(path, key_list, value):
    """
    update config yml
    :param key_list: yml key list
    :param path: yml path
    :param value: corresponding value
    :return:
    """
    config = get_config(path, easy=False)

    current_level = config
    outer_key = key_list[0]
    inner_key = key_list[1]
    if outer_key not in current_level:
        print(f'Update config Error: outermost key {outer_key} not exist!')
        exit()
    if inner_key not in current_level[outer_key]:
        print(f'Update config Error: inner key {inner_key} not exist in {outer_key}!')
        exit()

    current_level[outer_key][inner_key] = value

    with open(path, 'w') as f_writer:
        yaml.dump(config, f_writer, default_flow_style=False)
        f_writer.close()

def get_time_str():
    return time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())

def smart_tokenizer_and_embedding_resize(special_tokens_dict, tokenizer, model):
    """Resize tokenizer and embedding.

    Note: This is the unoptimized version that may make your embedding size not be divisible by 64.
    """
    num_new_tokens = tokenizer.add_special_tokens(special_tokens_dict)
    model.resize_token_embeddings(len(tokenizer))

    if num_new_tokens > 0:
        input_embeddings = model.get_input_embeddings().weight.data
        output_embeddings = model.get_output_embeddings().weight.data

        input_embeddings_avg = input_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)
        output_embeddings_avg = output_embeddings[:-num_new_tokens].mean(dim=0, keepdim=True)

        input_embeddings[-num_new_tokens:] = input_embeddings_avg
        output_embeddings[-num_new_tokens:] = output_embeddings_avg

def setup_ddp(rank, world_size, port):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = str(port)
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def find_free_port():
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as s:
        s.bind(('', 0))
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        return s.getsockname()[1]

def save_checkpoint(model, args, epoch=None):
    if args.stage1:
        path = args.dataset
        print(f"***** Saving model to {path}. *****")
        os.makedirs(path, exist_ok=True)
        # torch.save(model.stage1_model.state_dict(), os.path.join(path, f"mclp.pt"))
        loc_embedding = model.stage1_model.embedding_layer.location_embedding.weight
        time_embedding = model.stage1_model.embedding_layer.timeslot_embedding.weight
        user_embedding = model.stage1_model.embedding_layer.user_embedding.weight
        torch.save(loc_embedding, os.path.join(path, f"loc_embedding.pt"))
        torch.save(time_embedding, os.path.join(path, f"time_embedding.pt"))
        torch.save(user_embedding, os.path.join(path, f"user_embedding.pt"))
    if args.stage2:
        path = f'{args.dataset}/{args.llm}'
        os.makedirs(path, exist_ok=True)
        print(f"***** Saving model to {path}. *****")
        if args.lora:
            lora_parameters = peft.get_peft_model_state_dict(model.llm.model)
            torch.save(lora_parameters, os.path.join(path, f"lora_{epoch}.pt"))
        if args.align:
            torch.save(model.align_layer_loc.state_dict(), os.path.join(path, f'align_loc_{epoch}.pt'))
            torch.save(model.align_layer_time.state_dict(), os.path.join(path, f'align_time_{epoch}.pt'))
            torch.save(model.align_layer_user.state_dict(), os.path.join(path, f'align_user_{epoch}.pt'))
        if args.pre:
            torch.save(model.pre_layer.state_dict(), os.path.join(path, f'pre_{epoch}.pt'))
        torch.save(model.loc_header.state_dict(), os.path.join(path, f'predictor_{epoch}.pt'))


def modify_token_emb(tokenizer, inputs_ids, tokens_emb, data):
    if 'his_emb' in data.keys():
        his_emb = data['his_emb']
        cur_emb = data['cur_emb']
    if 'user_pre' in data.keys():
        user_pre = data['user_pre']
    his_token_id = tokenizer('[HisEmb]', return_tensors='pt', add_special_tokens=False)['input_ids'].item()
    cur_token_id = tokenizer('[CurEmb]', return_tensors='pt', add_special_tokens=False)['input_ids'].item()
    pre_token_id = tokenizer('[PreEmb]', return_tensors='pt', add_special_tokens=False)['input_ids'].item()

    for i, input_ids in enumerate(inputs_ids):

        idx_tensor_his = (input_ids == his_token_id).nonzero(as_tuple=True)[0]
        if idx_tensor_his.numel() > 0:
            for j, loc_emb in zip(idx_tensor_his, his_emb[i]):
                tokens_emb[i, j] = loc_emb

        idx_tensor_cur = (input_ids == cur_token_id).nonzero(as_tuple=True)[0]
        if idx_tensor_cur.numel() > 0:
            for j, loc_emb in zip(idx_tensor_cur, cur_emb[i]):
                tokens_emb[i, j] = loc_emb

        idx_tensor_pre = (input_ids == pre_token_id).nonzero(as_tuple=True)[0]
        if idx_tensor_pre.numel() > 0:
            original_emb = tokens_emb[i, idx_tensor_pre]
            tokens_emb[i, idx_tensor_pre] = user_pre[i].to(original_emb.dtype)

    return tokens_emb


def make_target_text(input_tokens, output_tokens, tokenizer):
    inputs_len = []
    input_ids = input_tokens['input_ids']
    input_attns = input_tokens['attention_mask']
    output_ids = output_tokens['input_ids']
    output_attns = output_tokens['attention_mask']
    bos_id = tokenizer.bos_token_id
    tokens = {"input_ids": [], "attention_mask": []}
    for i in range(len(input_ids)):
        mask_len = torch.sum(input_attns[i])
        inputs_len.append(mask_len)
        slice_idx = 0
        idx_tensor_bos = (output_ids[i] == bos_id).nonzero(as_tuple=True)[0]
        if idx_tensor_bos.numel() > 0:
            slice_idx = 1
        tokens['input_ids'].append(
            torch.cat([
                input_ids[i][:mask_len],
                output_ids[i][slice_idx:],
                input_ids[i][mask_len:]
            ])
        )
        tokens['attention_mask'].append(
            torch.cat([
                input_attns[i][:mask_len],
                output_attns[i][slice_idx:],
                input_attns[i][mask_len:]
            ])
        )
    tokens['input_ids'] = torch.stack(tokens['input_ids'])
    tokens['attention_mask'] = torch.stack(tokens['attention_mask'])

    return tokens, inputs_len


def load_checkpoint(model, args, device):
    path = f'{args.dataset}/{args.llm}'
    if args.stage1:
        stage1_model = torch.load(path + '/mclp.pt', map_location=device)
        model.stage1_model.load_state_dict(stage1_model)
        del stage1_model
    if args.stage2:
        if args.align:
            loc_align_parameters = torch.load(path + f'/align_loc_{args.epoch}.pt', map_location=device)
            time_align_parameters = torch.load(path + f'/align_time_{args.epoch}.pt', map_location=device)
            user_align_parameters = torch.load(path + f'/align_user_{args.epoch}.pt', map_location=device)
            model.align_layer_loc.load_state_dict(loc_align_parameters)
            model.align_layer_time.load_state_dict(time_align_parameters)
            model.align_layer_user.load_state_dict(user_align_parameters)
            del loc_align_parameters, time_align_parameters, user_align_parameters
        parameters = torch.load(path + f'/predictor_{args.epoch}.pt', map_location=device)
        model.loc_header.load_state_dict(parameters)
        del parameters
        if args.pre:
            parameters = torch.load(path + f'/pre_{args.epoch}.pt', map_location=device)
            model.pre_layer.load_state_dict(parameters)
            del parameters
        if args.lora:
            lora_path = os.path.join(args.dataset, args.llm, f'lora_{args.epoch}.pt')
            print(f'Loading LORA from {lora_path}')
            lora_parameters = torch.load(lora_path, map_location=device)
            peft.set_peft_model_state_dict(model.llm.model, lora_parameters)


def get_config(path, easy):
    """
    get config yml
    :param easy: is easydict mode
    :param path: yml path
    :return: EasyDict format
    """
    f = open(path, 'r', encoding='utf-8')
    res = yaml.safe_load(f)
    if easy:
        return EasyDict(res)
    else:
        return res

# Function to randomly sample a subset of the dataset
def sample_subset(dataset, num_samples):
    indices = list(range(len(dataset)))
    random.shuffle(indices)
    sampled_indices = indices[:num_samples]
    return Subset(dataset, sampled_indices)

def init_tokenizer(model_id, additional_tokens=None):
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)
    special_tokens_dict = {}

    if tokenizer.pad_token is None:
        special_tokens_dict["pad_token"] = "[PAD]"
    if tokenizer.eos_token is None:
        special_tokens_dict["eos_token"] = "</s>"
    if tokenizer.bos_token is None:
        special_tokens_dict["bos_token"] = "<s>"
    if tokenizer.unk_token is None:
        special_tokens_dict["unk_token"] = "<UNK>"

    if additional_tokens:
        special_tokens_dict['additional_special_tokens'] = additional_tokens

    tokenizer.add_special_tokens(special_tokens_dict)
    return tokenizer


def init_llama(model_id, tokenizer):
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map='auto'
    )
    model.resize_token_embeddings(len(tokenizer))

    return model