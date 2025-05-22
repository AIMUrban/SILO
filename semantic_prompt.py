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

from collections import OrderedDict
import numpy as np
import pandas as pd
import torch
import tqdm
from tools import init_tokenizer, init_llama, get_mapper


def map_to_period(t):
    hour = (t // 2) % 24
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 24:
        return "evening"
    else:
        return "night"


def map_to_weekday(d):
    weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    return weekdays[d % 7]

def map_to_weekday_period(t, d):
    weekday = d % 7
    hour = (t // 2) % 24
    if 6 <= hour < 12:
        period = "morning"
    elif 12 <= hour < 18:
        period = "afternoon"
    elif 18 <= hour < 24:
        period = "evening"
    else:
        period = "night"
    if weekday in {0, 6}:
        day = 'Weekend'
    else:
        day = 'Weekday'
    return f"{day} {period}"

def compute_user_embeddings(train_data, time_slots_key, tokenizer, model, args):
    train_data['period'] = train_data['t'].apply(map_to_period)
    train_data['weekday'] = train_data['d'].apply(map_to_weekday)
    # train_data['wp'] = train_data.apply(lambda row: map_to_weekday_period(row['t'], row['d']), axis=1)
    train_data['time_slot'] = train_data.apply(lambda row: f"{row['weekday']} {row['period']}", axis=1)
    grouped = train_data.groupby(['user_id', 'time_slot']).size().reset_index(name='count')

    result = []

    for user_id, user_data in tqdm.tqdm(grouped.groupby('user_id'), desc=f'Fetching semantic embeddings'):
        formatted_data = OrderedDict((time_key, 0) for time_key in time_slots_key)

        user_data = grouped[grouped['user_id'] == user_id]

        for _, row in user_data.iterrows():
            formatted_data[row['time_slot']] = row['count']

        total_count = sum(formatted_data.values())
        formatted_data = {k: v / total_count for k, v in formatted_data.items()}

        instruction = (f"Given a user's historical activity occurrence data in JSON format, "
                       f"where each key indicates a specific time period (e.g., Sunday morning), "
                       f"and each value represents how frequently the user is active during that time. ")

        user_occurrence = ','.join([f"'{k}':{v:.{args.numf_l}f}" for k, v in formatted_data.items()])
        prompt = f"{instruction}{{{user_occurrence}}}. Please summarize this user."

        if user_id == list(grouped['user_id'].unique())[0]:
            print(f"Prompt sample:\n{prompt}")
        if user_id == list(grouped['user_id'].unique())[-10]:
            print(f"Prompt sample:\n{prompt}")

        text_input_tokens = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=1024
        )
        input_embeds = model.get_input_embeddings()(text_input_tokens['input_ids'].to('cuda'))

        with torch.no_grad():
            output = model(
                inputs_embeds=input_embeds,
                output_hidden_states=True,
            )
        embedding = output['hidden_states'][-1].mean(dim=1)
        result.append(embedding.cpu())

    return torch.cat(result)


def compute_location_embeddings(train_data, location2id, time_slots_key, tokenizer, model, args):
    train_data['period'] = train_data['t'].apply(map_to_period)
    train_data['weekday'] = train_data['d'].apply(map_to_weekday)
    # train_data['wp'] = train_data.apply(lambda row: map_to_weekday_period(row['t'], row['d']), axis=1)
    train_data['time_slot'] = train_data.apply(lambda row: f"{row['weekday']} {row['period']}", axis=1)
    print(train_data.head())
    grouped = train_data.groupby(['loc_id', 'time_slot']).size().reset_index(name='count')
    # print(grouped)
    # exit()

    locid2timeslot_count = {}
    for lid, df_g in grouped.groupby('loc_id'):
        tmp_dict = {}
        for _, row in df_g.iterrows():
            tmp_dict[row['time_slot']] = row['count']
        locid2timeslot_count[lid] = tmp_dict

    all_locs = sorted(location2id.values())

    result = []

    for i, loc_id in tqdm.tqdm(enumerate(all_locs), total=len(all_locs), desc="Generating location embeddings"):

        formatted_data = OrderedDict((time_key, 0) for time_key in time_slots_key)

        if loc_id in locid2timeslot_count:
            timeslot_count_dict = locid2timeslot_count[loc_id]
            for k, v in timeslot_count_dict.items():
                formatted_data[k] = v

        total_count = sum(formatted_data.values())
        frequencies = {k: v / (total_count+1e-9) for k, v in formatted_data.items()}

        instruction = (
            "Given the historical visitation data of a location in JSON format, "
            "where each key indicates a specific time period (e.g., Sunday morning), "
            "and each value represents how frequently the location is visited during that time. "
        )

        location_occurrence = ','.join([f"'{k}':{v:.{args.numf_l}f}" for k, v in frequencies.items()])
        prompt = f"{instruction}{{{location_occurrence}}}. Please summarize this location."

        if loc_id == list(grouped['loc_id'].unique())[0]:
            print(f"Prompt sample:\n{prompt}")
        if loc_id == list(grouped['loc_id'].unique())[110]:
            print(f"Prompt sample:\n{prompt}")
        # if loc_id not in list(grouped['loc_id'].unique()):
        #     print(f"Prompt sample:\n{prompt}")

        text_input_tokens = tokenizer(
            prompt,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=1024
        )
        input_embeds = model.get_input_embeddings()(text_input_tokens['input_ids'].to('cuda'))

        with torch.no_grad():
            output = model(
                inputs_embeds=input_embeds,
                output_hidden_states=True,
            )
        embedding = output['hidden_states'][-1].mean(dim=1)
        result.append(embedding.cpu())

    return torch.cat(result)

def get_sem_emb(args):
    dataset_path = args.dataset
    get_mapper(dataset_path)

    train_data = pd.read_parquet(os.path.join(dataset_path, 'train_sampled.parquet'))
    user2id = np.load(os.path.join(dataset_path, 'user_mapper.npy'), allow_pickle=True).item()
    location2id = np.load(os.path.join(dataset_path, 'location_mapper.npy'), allow_pickle=True).item()
    if 'x' in train_data.columns and 'y' in train_data.columns:
        train_data['loc'] = train_data.apply(lambda row: f"{int(row['x'])}_{int(row['y'])}", axis=1)
    else:
        train_data['loc'] = train_data['loc'].astype(str)
    train_data['user_id'] = train_data['uid'].map(user2id)
    train_data['loc_id'] = train_data['loc'].map(location2id)

    model_id = args.llm_id
    print(f'***** Current LLM: {model_id} *****')
    tokenizer = init_tokenizer(model_id=model_id)
    model = init_llama(model_id, tokenizer)

    time_slots_key = [
        f"{day} {period}" for day in
        ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        for period in ["morning", "afternoon", "evening", "night"]
    ]

    print(f'***** Current dataset: {dataset_path} *****')
    print("***** Hidden size:", model.config.hidden_size, '*****')
    save_path = os.path.join(dataset_path, args.llm)
    os.makedirs(save_path, exist_ok=True)

    location_embeddings = compute_location_embeddings(train_data, location2id, time_slots_key, tokenizer, model, args)
    print(location_embeddings.shape)
    torch.save(location_embeddings, os.path.join(str(save_path), f"loc_sem_embeddings_{args.numf_l}f.pt"))

    user_embeddings = compute_user_embeddings(train_data, time_slots_key, tokenizer, model, args)
    print(user_embeddings.shape)
    torch.save(user_embeddings, os.path.join(str(save_path), f"user_sem_embeddings_{args.numf_l}f.pt"))

get_sem_emb(args_sys)
