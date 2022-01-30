import os
import torch
import random
import pandas as pd
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
from models import model_attributes
from torch.utils.data import Dataset, Subset

class ConfounderDataset(Dataset):
    def __init__(self, root_dir,
                 target_name, confounder_names,
                 model_type=None, augment_data=None):
        raise NotImplementedError

    def __len__(self):
        return len(self.filename_array)

    def __getitem__(self, idx):
        y = self.y_array[idx]
        g = self.group_array[idx]

        if model_attributes[self.model_type]['feature_type']=='precomputed':
            x = self.features_mat[idx, :]
        else:
            img_filename = os.path.join(
                self.data_dir,
                self.filename_array[idx])
            img = Image.open(img_filename).convert('RGB')
            # Figure out split and transform accordingly
            if self.split_array[idx] == self.split_dict['train'] and self.train_transform:
                img = self.train_transform(img)
            elif (self.split_array[idx] in [self.split_dict['val'], self.split_dict['test']] and
              self.eval_transform):
                img = self.eval_transform(img)
            # Flatten if needed
            if model_attributes[self.model_type]['flatten']:
                assert img.dim()==3
                img = img.view(-1)
            x = img

            # TODO: Include mixup logic here -- only if the given example is a training example
            if self.split_array[idx] == self.split_dict['train'] and self.mixup:
                # Select a random number from the given beta distribution
                alpha = 0.2
                lam = np.random.beta(alpha, alpha)
                
                pseudo_mixup = not self.complete_mixup  # Mixup only examples from the same class
                if pseudo_mixup:
                    # print("Using pseudo mixup (mixing examples only from the same class)...")
                    
                    # Choose another image randomly from the same class
                    same_class_instances = [i for i in range(len(self.filename_array)) if self.y_array[i] == y]
                    mixup_idx = np.random.choice(same_class_instances)
                    mixup_img_filename = os.path.join(
                                    self.data_dir,
                                    self.filename_array[mixup_idx])
                    mixup_img = Image.open(mixup_img_filename).convert('RGB')

                    if self.train_transform:
                        mixup_img = self.train_transform(mixup_img)

                    # Flatten if needed
                    if model_attributes[self.model_type]['flatten']:
                        assert mixup_img.dim()==3
                        mixup_img = mixup_img.view(-1)

                    # Mixup the images from the same class
                    x = lam * img + (1 - lam) * mixup_img
                
                else:
                    # print("Using proper mixup...")
                    
                    # Choose another image/label randomly
                    mixup_idx = random.randint(0, len(self.filename_array)-1)
                    mixup_y = torch.zeros(10)
                    mixup_y[self.y_array[mixup_idx]] = 1.
                    mixup_img_filename = os.path.join(
                                    self.data_dir,
                                    self.filename_array[mixup_idx])
                    mixup_img = Image.open(mixup_img_filename).convert('RGB')

                    if self.train_transform:
                        mixup_img = self.train_transform(mixup_img)

                    # Flatten if needed
                    if model_attributes[self.model_type]['flatten']:
                        assert mixup_img.dim()==3
                        mixup_img = mixup_img.view(-1)

                    # Mix the two samples including both the images as well as the labels
                    x = lam * img + (1 - lam) * mixup_img
                    onehot_y = torch.zeros(10)
                    onehot_y[y] = 1.
                    y = lam * onehot_y + (1 - lam) * mixup_y

        return x,y,g

    def get_splits(self, splits, train_frac=1.0, subsample_to_minority=False):
        subsets = {}
        for split in splits:
            assert split in ('train','val','test'), split+' is not a valid split'
            split_mask = self.split_array == self.split_dict[split]
            num_split = np.sum(split_mask)
            indices = np.where(split_mask)[0]
            if split == 'train':
                if subsample_to_minority:
                    group_counts = (np.arange(self.n_groups).reshape(-1, 1)==self.group_array[indices]).sum(1)
                    smallest_group_size = np.min(group_counts)
                    indices = np.array([], dtype=int)
                    for g in np.arange(self.n_groups):
                        group_indices = np.where((self.group_array == g) & split_mask)[0]                        
                        indices = np.concatenate((
                            indices, np.sort(np.random.permutation(group_indices)[:smallest_group_size])))
                if train_frac<1:
                    num_to_keep = int(np.round(float(len(indices)) * train_frac))
                    indices = np.sort(np.random.permutation(indices)[:num_to_keep])
            subsets[split] = Subset(self, indices)
        return subsets

    def group_str(self, group_idx):
        y = group_idx // (self.n_groups/self.n_classes)
        c = group_idx % (self.n_groups//self.n_classes)

        group_name = f'{self.target_name} = {int(y)}'
        bin_str = format(int(c), f'0{self.n_confounders}b')[::-1]
        for attr_idx, attr_name in enumerate(self.confounder_names):
            group_name += f', {attr_name} = {bin_str[attr_idx]}'
        return group_name
