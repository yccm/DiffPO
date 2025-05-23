import pickle
import yaml
import os
import re
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader, Dataset


def process_func(path: str, aug_rate=1, missing_ratio=0.1, train=True, dataset_name = 'acic', current_id='0'):
    data = pd.read_csv(path, sep = ',', decimal = ',')
    data.replace("?", np.nan, inplace=True)
    data_aug = pd.concat([data] * aug_rate)

    observed_values = data_aug.values.astype("float32")
    observed_masks = ~np.isnan(observed_values)

    masks = observed_masks.copy()

    if dataset_name == 'acic2016':
        load_mask_path = "./data_acic2016/acic2016_mask/" + current_id + ".csv"
        print(load_mask_path)

    if dataset_name == 'acic2018':
        load_mask_path = "./data_acic2018/acic2018_mask/" + current_id + ".csv"
        print(load_mask_path)

    load_mask = pd.read_csv(load_mask_path, sep = ',', decimal = ',')
    load_mask = load_mask.values.astype("float32")

    if train:
        gt_masks = load_mask

        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_masks.astype(int)
        gt_masks = gt_masks.astype(int)

    else:
        gt_masks = load_mask
        gt_masks[:, 1] = 0
        gt_masks[:, 2] = 0
        
        observed_values = np.nan_to_num(observed_values)
        observed_masks = observed_masks.astype(int)
        gt_masks = gt_masks.astype(int)

    return observed_values, observed_masks, gt_masks


class tabular_dataset(Dataset):
    def __init__(
        self, eval_length=100, use_index_list=None, aug_rate=1, missing_ratio=0.1, seed=0, train=True, dataset_name = 'acic', current_id='0'
    ):  
        if dataset_name == 'acic2016':
            self.eval_length = 87
        if dataset_name == 'acic2018':
            self.eval_length = 182

        np.random.seed(seed)

        if dataset_name == 'acic2016':
            dataset_path = "./data_acic2016/acic2016_norm_data/" + current_id + ".csv"
            print('dataset_path', dataset_path)
            processed_data_path = (
                f"./data_acic2016/missing_ratio-{missing_ratio}_seed-{seed}.pk"
            )
            processed_data_path_norm = (
                f"./data_acic2016/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
            )
            os.system('rm {}'.format(processed_data_path))

        if dataset_name == 'acic2018':
            dataset_path = "./data_acic2018/acic2018_norm_data/" + current_id + ".csv"
            print('dataset_path', dataset_path)
            processed_data_path = (
                f"./data_acic2018/missing_ratio-{missing_ratio}_seed-{seed}.pk"
            )
            processed_data_path_norm = (
                f"./data_acic2018/missing_ratio-{missing_ratio}_seed-{seed}_max-min_norm.pk"
            )
            os.system('rm {}'.format(processed_data_path))

        if not os.path.isfile(processed_data_path):
            self.observed_values, self.observed_masks, self.gt_masks = process_func(
                dataset_path, aug_rate=aug_rate, missing_ratio=missing_ratio, train=train, dataset_name=dataset_name, current_id=current_id
            )

            with open(processed_data_path, "wb") as f:
                pickle.dump(
                    [self.observed_values, self.observed_masks, self.gt_masks], f
                )
            print("--------Dataset created--------")

        elif os.path.isfile(processed_data_path_norm):
            with open(processed_data_path_norm, "rb") as f:
                self.observed_values, self.observed_masks, self.gt_masks = pickle.load(
                    f
                )
            print("--------Normalized dataset loaded--------")

        if use_index_list is None:
            self.use_index_list = np.arange(len(self.observed_values))
        else:
            self.use_index_list = use_index_list

    def __getitem__(self, org_index):
        index = self.use_index_list[org_index]
        s = {
            "observed_data": self.observed_values[index],
            "observed_mask": self.observed_masks[index],
            "gt_mask": self.gt_masks[index],
            "timepoints": np.arange(self.eval_length),
        }
        return s

    def __len__(self):
        return len(self.use_index_list)


def get_dataloader(seed=1, nfold=5, batch_size=16, missing_ratio=0.1, dataset_name = 'acic2018', current_id='0'):
    dataset = tabular_dataset(missing_ratio=missing_ratio, seed=seed, dataset_name = dataset_name, current_id=current_id)
    print(f"Dataset size:{len(dataset)} entries")

    indlist = np.arange(len(dataset))

    if dataset_name == 'acic2016':
        
        total_samples = len(indlist)
        
        # Randomly shuffle all indices
        shuffled_indices = np.arange(total_samples)
        np.random.shuffle(shuffled_indices)
        
        train_ratio = 0.7
        valid_ratio = 0.1
        test_ratio = 0.2
        
        num_train = int(total_samples * train_ratio)
        num_valid = int(total_samples * valid_ratio)
        num_test = total_samples - num_train - num_valid
        
        train_index = shuffled_indices[:num_train]
        valid_index = shuffled_indices[num_train:num_train + num_valid]
        test_index = shuffled_indices[num_train + num_valid:]
        
        processed_data_path_norm = (
            f"./data_acic2016/missing_ratio-{missing_ratio}_seed-{seed}_current_id-{current_id}_max-min_norm.pk"
        )
        # if not os.path.isfile(processed_data_path_norm):
        # normalize anyway.
        print(
            "------------- Perform data normalization and store the mean value of each column.--------------"
    )
    # data transformation after train-test split.

    if dataset_name == 'acic2018':
        total_samples = len(indlist)
        
        shuffled_indices = np.arange(total_samples)
        np.random.shuffle(shuffled_indices)
        
        train_ratio = 0.7
        valid_ratio = 0.1
        test_ratio = 0.2
        
        num_train = int(total_samples * train_ratio)
        num_valid = int(total_samples * valid_ratio)
        num_test = total_samples - num_train - num_valid
        
        train_index = shuffled_indices[:num_train]
        valid_index = shuffled_indices[num_train:num_train + num_valid]
        test_index = shuffled_indices[num_train + num_valid:]
        
        processed_data_path_norm = (
            f"./data_acic2018/missing_ratio-{missing_ratio}_seed-{seed}_current_id-{current_id}_max-min_norm.pk"
        )
        # if not os.path.isfile(processed_data_path_norm):
        # normalize anyway.
        print(
            "------------- Perform data normalization and store the mean value of each column.--------------"
        )
        # data transformation after train-test split.

    col_num = dataset.observed_values.shape[1]
    max_arr = np.zeros(col_num)
    min_arr = np.zeros(col_num)
    mean_arr = np.zeros(col_num)

    with open(processed_data_path_norm, "wb") as f:
        pickle.dump(
            [dataset.observed_values, dataset.observed_masks, dataset.gt_masks], f
        )

    train_dataset = tabular_dataset(
        use_index_list=train_index, missing_ratio=missing_ratio, seed=seed, dataset_name = dataset_name, current_id=current_id
    )
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=1)

    valid_dataset = tabular_dataset(
        use_index_list=valid_index, missing_ratio=missing_ratio, seed=seed, train=False, dataset_name = dataset_name, current_id=current_id
    )
    valid_loader = DataLoader(valid_dataset, batch_size=batch_size, shuffle=0)

    test_dataset = tabular_dataset(
        use_index_list=test_index, missing_ratio=missing_ratio, seed=seed, train=False, dataset_name = dataset_name, current_id=current_id
    )
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=0)

    print(f"Training dataset size: {len(train_dataset)}")
    print(f"Testing dataset size: {len(test_dataset)}")

    return train_loader, valid_loader, test_loader