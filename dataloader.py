import json
import os
from collections import OrderedDict

import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from tqdm import tqdm

from tools import get_mapper


def map_to_period(timeslot):
    hour = (timeslot // 2) % 24
    if 6 <= hour < 12:
        return "morning"
    elif 12 <= hour < 18:
        return "afternoon"
    elif 18 <= hour < 24:
        return "evening"
    else:
        return "night"

def map_to_4x7_continuous(weekday, timeslot):
    hour = (timeslot // 2)
    if 6 <= hour < 12:
        period = 0  # Morning
    elif 12 <= hour < 18:
        period = 1  # Afternoon
    elif 18 <= hour < 24:
        period = 2  # Evening
    else:
        period = 3  # Night

    return weekday * 4 + period

def map_to_weekday(d):
    weekdays = ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
    return weekdays[d % 7]

def get_user_json(dataset_path, args):
    time_slots_key = [
        f"{day} {period}" for day in
        ["Sunday", "Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday"]
        for period in ["morning", "afternoon", "evening", "night"]
    ]

    train_data = pd.read_parquet(os.path.join(dataset_path, 'train_sampled.parquet'))
    user2id = np.load(os.path.join(dataset_path, 'user_mapper.npy'), allow_pickle=True).item()
    location2id = np.load(os.path.join(dataset_path, 'location_mapper.npy'), allow_pickle=True).item()

    train_data['loc'] = train_data.apply(lambda row: f"{row['x']}_{row['y']}", axis=1)
    train_data['user_id'] = train_data['uid'].map(user2id)
    train_data['loc_id'] = train_data['loc'].map(location2id)

    train_data['period'] = train_data['t'].apply(map_to_period)
    train_data['weekday'] = train_data['d'].apply(map_to_weekday)
    train_data['time_slot'] = train_data.apply(lambda row: f"{row['weekday']} {row['period']}", axis=1)
    grouped = train_data.groupby(['user_id', 'time_slot']).size().reset_index(name='count')
    result = {}
    i = 0
    for user_id, user_data in tqdm(grouped.groupby('user_id'), desc=f'Fetching occurrence'):
        formatted_data = OrderedDict((time_key, 0) for time_key in time_slots_key)
        user_data = grouped[grouped['user_id'] == user_id]
        for _, row in user_data.iterrows():
            formatted_data[row['time_slot']] = row['count']
        total_count = sum(formatted_data.values())
        formatted_data = {k: v / total_count for k, v in formatted_data.items()}
        user_occurrence = ','.join([f"'{k}':{v:.{args.numf_u}f}" for k, v in formatted_data.items()])
        result[user_id] = "{"+user_occurrence+"}"
        if i == 100:
            print(f"Sample:\n{result[user_id]}")
        if i == 299:
            print(f"Sample:\n{result[user_id]}")
        i += 1

    return result


def generate_data(args, load_mode):
    dataset_path = args.dataset
    train_historical_file = os.path.join(dataset_path, 'train_historical.parquet')

    if os.path.exists(os.path.join(dataset_path, f'{load_mode}_{args.numf_u}f.npy')):
        print(f'{load_mode}.npy already exists, skipping.''')
        return

    get_mapper(dataset_path)
    user2id = np.load(os.path.join(dataset_path, 'user_mapper.npy'), allow_pickle=True).item()
    location2id = np.load(os.path.join(dataset_path, 'location_mapper.npy'), allow_pickle=True).item()

    historical_len = 35
    current_len = 6
    window_step = 2
    traj_length = historical_len+current_len

    data_file = 'train_sampled.parquet' if 'train' in load_mode else 'test_sampled.parquet'
    data_res = []
    data = pd.read_parquet(os.path.join(dataset_path, data_file))

    if 'test' in load_mode:
        train_data = pd.read_parquet(os.path.join(dataset_path, 'train_sampled.parquet'))
        historical_data = train_data.groupby('uid').apply(lambda x: x.iloc[-(traj_length-1):]).reset_index(drop=True)
        historical_data.to_parquet(train_historical_file)
        historical_data = historical_data.sort_values(by=['uid', 'd', 't']).reset_index(drop=True)
        data = pd.concat([historical_data, data]).sort_values(by=['uid', 'd', 't']).reset_index(drop=True)

    data['continuous_t'] = data['d'] * 48 + data['t']
    data['weekday'] = data['d'] % 7

    data['loc'] = data.apply(lambda row: f"{row['x']}_{row['y']}", axis=1)

    data['user_id'] = data['uid'].map(user2id)
    data['loc_id'] = data['loc'].map(location2id)
    data['timeslot'] = data['t'] // 2

    user_json = get_user_json(dataset_path, args)

    for user_id in tqdm(data['user_id'].unique(), desc=f'creating {load_mode} trajectories'):
        user_data = data[data['user_id'] == user_id].sort_values(by=['d', 't']).reset_index(drop=True)

        user_data['timeslot_str'] = user_data['t'].apply(map_to_period)
        user_data['weekday_str'] = user_data['d'].apply(map_to_weekday)
        user_data['continuous_4x7'] = user_data.apply(lambda row: map_to_4x7_continuous(row['weekday'], row['timeslot']), axis=1)

        for start in range(0, len(user_data) - traj_length + 1, window_step):
            traj = user_data.iloc[start:start + traj_length]
            traj_data = {
                'user': traj['user_id'].iloc[0],
                'loc_his': traj['loc_id'].values[:historical_len],
                'loc_cur': traj['loc_id'].values[historical_len:-1],
                'timeslot_his': traj['continuous_4x7'].values[:historical_len],
                'timeslot_cur': traj['continuous_4x7'].values[historical_len:-1],
                'loc_label': traj['loc_id'].values[-1],
            }
            extra_dict = {
                'time_str_his': traj['timeslot_str'].values[:historical_len],
                'time_str_cur': traj['timeslot_str'].values[historical_len:-1],
                'weekday_str_his': traj['weekday_str'].values[:historical_len],
                'weekday_str_cur': traj['weekday_str'].values[historical_len:-1],
                'user_json': user_json[user_id]
            }
            traj_data.update(extra_dict)
            data_res.append(traj_data)

    np.save(os.path.join(dataset_path, f'{load_mode}_{args.numf_u}f.npy'), data_res)

def load_npy_file(save_path):
    loaded_data = np.load(save_path, allow_pickle=True)
    return loaded_data

class MyDataset(Dataset):
    def __init__(self, args, load_mode):
        self.load_mode = load_mode
        dataset_path = args.dataset

        self.user2id = np.load(os.path.join(dataset_path, 'user_mapper.npy'), allow_pickle=True).item()
        self.location2id = np.load(os.path.join(dataset_path, 'location_mapper.npy'), allow_pickle=True).item()

        self.data = load_npy_file(os.path.join(dataset_path, f'{load_mode}_{args.numf_u}f.npy'))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
