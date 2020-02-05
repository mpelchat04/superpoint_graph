#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Jan 29 15:21 2020
@author: mpelchat04
"""
from __future__ import division
from __future__ import print_function
from builtins import range

import random
import numpy as np
import os
import functools
import torch
import torchnet as tnt
import h5py
import learning.spg as spg
import warnings
import supervized_partition.graph_processing_custom as graph_processing


def get_datasets(args, test_seed_offset=0):
    """ Gets training and test datasets. """

    # Load superpoints graphs
    # List all .las files in the folder
    root = args.AIRBORNE_LIDAR_PATH

    if args.dataset == 'airborne_lidar':
        folders = ["trn", "val", "tst"]
    else:
        raise ValueError('%s is an unknown data set' % args.dataset)

    dataset_dict = {'trn': [], 'val': [], 'tst': []}
    for folder in folders:
        path_spg = f"{root}/superpoint_graphs/{folder}/"
        if not os.path.isdir(f"{path_spg}/"):
            raise ValueError(f"{path_spg} does not exist.")

        for fname in sorted(os.listdir(path_spg)):
            if fname.endswith(".h5"):
                dataset_dict[folder].append(spg.spg_reader(args=args, fname=(path_spg + fname), incl_dir_in_name=True))

    # Normalize Edge features.
    if args.spg_attribs01:
        dataset_dict, scaler = spg.scaler_custom(dataset_dict, transform_train=True)
    else:
        warnings.warn(f"Normalize Edge attributes not set to True. Scaler will be set to None.")
        scaler = None

    trn_dataset = tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in dataset_dict['trn']],
                                          functools.partial(spg.loader, train=True, args=args, db_path=root))
    val_dataset = tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in dataset_dict['val']],
                                          functools.partial(spg.loader, train=True, args=args, db_path=root, test_seed_offset=test_seed_offset))
    tst_dataset = tnt.dataset.ListDataset([spg.spg_to_igraph(*tlist) for tlist in dataset_dict['tst']],
                                          functools.partial(spg.loader, train=True, args=args, db_path=root, test_seed_offset=test_seed_offset))

    return trn_dataset, tst_dataset, val_dataset, scaler


def get_info(args):
    edge_feats = 0
    info_dataset_dict = graph_processing.get_airborne_lidar_info()
    for attrib in args.edge_attribs.split(','):
        # default edge_attribs are all the attribs in the paper (13 of them).
        a = attrib.split('/')[0]
        if a in ['delta_avg', 'delta_std', 'xyz']:
            edge_feats += 3
        else:
            edge_feats += 1
    if args.loss_weights == 'none':
        # Weight for every class. Hence the 4 for airborne_lidar.
        weights = np.ones((info_dataset_dict['classes'],), dtype='f4')
    else:
        weights = h5py.File(args.AIRBORNE_LIDAR_PATH + "/parsed/class_count.h5")["class_count"][:].astype('f4')
        weights = weights[:, [i for i in range(6) if i != args.cvfold - 1]].sum(1)
        weights = (weights + 1).mean() / (weights + 1)
    if args.loss_weights == 'sqrt':
        weights = np.sqrt(weights)
    weights = torch.from_numpy(weights).cuda() if args.cuda else torch.from_numpy(weights)

    return {
        'node_feats': 9 if args.pc_attribs == '' else len(args.pc_attribs),
        'edge_feats': edge_feats,
        'classes': info_dataset_dict['classes'],
        'class_weights': weights,
        'inv_class_map': info_dataset_dict['inv_class_map'],
    }


def preprocess_pointclouds(folder):
    """ Preprocesses data by splitting them by components and normalizing."""

    # folder = /wspace/disk01/lidar/POINTCLOUD
    class_count = np.zeros((4,), dtype='int')
    for dataset in ['trn', 'val', 'tst']:
        path_parsed = f"{folder}/parsed/{dataset}/"
        path_feat = f"{folder}/features/{dataset}/"
        path_spg = f"{folder}/superpoint_graphs/{dataset}/"

        if not os.path.exists(path_parsed):
            os.makedirs(path_parsed)
        random.seed(0)

        for file in os.listdir(path_spg):
            print(file)
            if file.endswith(".h5"):
                f = h5py.File(path_feat + file, 'r')
                xyz = f['xyz'][:]
                intensity = f['intensity'][:]
                labels = f['labels'][:]
                # hard_labels = np.argmax(labels[:, 1:], 1)
                label_count = np.bincount(labels[:, 0], minlength=4)
                class_count = class_count + label_count

                # Normalize intensity
                norm_intensity = (intensity - np.min(intensity)) / (np.max(intensity) - np.min(intensity))
                # Normalize elevation
                norm_elevation = (xyz[:, 2] - np.min(xyz[:, 2])) / (np.max(xyz[:, 2] - np.min(xyz[:, 2])))
                norm_elevation = norm_elevation.reshape((norm_elevation.shape[0], 1))
                # Normalize X and Y
                norm_x = (xyz[:, 0] - np.min(xyz[:, 0])) / (np.max(xyz[:, 0] - np.min(xyz[:, 0])))
                norm_x = norm_x.reshape((norm_x.shape[0], 1))
                norm_y = (xyz[:, 1] - np.min(xyz[:, 1])) / (np.max(xyz[:, 1] - np.min(xyz[:, 1])))
                norm_y = norm_y.reshape((norm_y.shape[0], 1))
                lspv = f["geof"][:]
                nb_return = f['nb_return'][:]
                parsed = np.concatenate([norm_x, norm_y, norm_elevation, norm_intensity, nb_return, lspv], axis=1)

                f = h5py.File(path_spg + file, 'r')
                numc = len(f['components'].keys())

                with h5py.File(path_parsed + file, 'w') as hf:
                    hf.create_dataset(name='centroid', data=xyz.mean(0))
                    for c in range(numc):
                        idx = f['components/{:d}'.format(c)][:].flatten()
                        if idx.size > 10000:  # trim extra large segments, just for speed-up of loading time
                            ii = random.sample(range(idx.size), k=10000)
                            idx = idx[ii]

                        hf.create_dataset(name='{:d}'.format(c), data=parsed[idx, ...])

    path = '{}/parsed/'.format(folder)
    data_file = h5py.File(path + 'class_count.h5', 'w')
    data_file.create_dataset('class_count', data=class_count, dtype='int')


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
    parser.add_argument('--ROOT_PATH', default='datasets/POINTCLOUD')
    args = parser.parse_args()
    preprocess_pointclouds(args.ROOT_PATH)
