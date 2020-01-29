"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
    Script for partioning into simples shapes
"""
import os.path
import sys
import numpy as np
import argparse
from timeit import default_timer as timer
import glob
import warnings

sys.path.append("./partition/cut-pursuit/build/src")
sys.path.append("./partition/ply_c")
sys.path.append("./partition")
import libcp
import libply_c
from graphs import *
from provider import *

parser = argparse.ArgumentParser(description='Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs')
parser.add_argument('--ROOT_PATH', default='datasets/s3dis')
parser.add_argument('--dataset', default='s3dis', help='s3dis/sema3d/your_dataset')
parser.add_argument('--k_nn_geof', default=45, type=int, help='number of neighbors for the geometric features')
parser.add_argument('--k_nn_adj', default=10, type=int, help='adjacency structure for the minimal partition')
parser.add_argument('--lambda_edge_weight', default=1., type=float, help='parameter determine the edge weight for minimal part.')
parser.add_argument('--reg_strength', default=0.1, type=float, help='regularization strength for the minimal partition')
parser.add_argument('--d_se_max', default=0, type=float, help='max length of super edges')
parser.add_argument('--voxel_width', default=0.03, type=float, help='voxel size when subsampling (in m)')
parser.add_argument('--ver_batch', default=0, type=int, help='Batch size for reading large files, 0 do disable batch loading')
parser.add_argument('--overwrite', default=0, type=int, help='Wether to read existing files or overwrite them')
args = parser.parse_args()

# path to data
root = args.ROOT_PATH + '/'
# list of subfolders to be processed
if args.dataset == 'airborne_lidar':
    folders = ["trn/", "val/", "tst/"]
    n_labels = 4
else:
    raise ValueError('%s is an unknown data set' % dataset)

times = [0, 0, 0]  # time for computing: features / partition / spg

if not os.path.isdir(root + "clouds"):
    os.mkdir(root + "clouds")
if not os.path.isdir(root + "features"):
    os.mkdir(root + "features")
if not os.path.isdir(root + "superpoint_graphs"):
    os.mkdir(root + "superpoint_graphs")

for folder in folders:
    print("=================\n   " + folder + "\n=================")

    data_folder = root + "data/" + folder
    cloud_folder = root + "clouds/" + folder
    fea_folder = root + "features/" + folder
    spg_folder = root + "superpoint_graphs/" + folder
    if not os.path.isdir(data_folder):
        raise ValueError("%s does not exist" % data_folder)

    if not os.path.isdir(cloud_folder):
        os.mkdir(cloud_folder)
    if not os.path.isdir(fea_folder):
        os.mkdir(fea_folder)
    if not os.path.isdir(spg_folder):
        os.mkdir(spg_folder)

    # List all .las files in the folder
    files = glob.glob(data_folder + "*.las")

    if len(files) == 0:
        warnings.warn(f"{data_folder} is empty")
        # raise ValueError('%s is empty' % data_folder)

    n_files = len(files)
    i_file = 0
    for file in files:
        file_name = os.path.splitext(os.path.basename(file))[0]

        # Define the 4 required files.
        data_file = data_folder + file_name + '.las'
        cloud_file = cloud_folder + file_name
        fea_file = fea_folder + file_name + '.h5'
        spg_file = spg_folder + file_name + '.h5'

        i_file = i_file + 1
        print(str(i_file) + " / " + str(n_files) + "---> " + file_name)
        # --- build the geometric feature file h5 file ---
        if os.path.isfile(fea_file) and not args.overwrite:
            print("    reading the existing feature file...")
            geof, xyz, rgb, graph_nn, labels, intensity, nb_return = read_features(fea_file)
        else:
            print("    creating the feature file...")
            # --- read the data files and compute the labels---
            xyz, nb_return, intensity, labels = read_airborne_lidar_format(data_file)
            # if no rgb available simply set here rgb = [] and make sure to not use it later on
            rgb = []
            if args.voxel_width > 0:
                warnings.warn(f"Pruning is not implemented with the dataset {args.dataset}. The whole dataset will be used.")

            start = timer()
            # ---compute 10 nn graph-------
            graph_nn, target_fea = compute_graph_nn_2(xyz, args.k_nn_adj, args.k_nn_geof)

            # ---compute geometric features-------
            # Geometric features are linearity, planarity, scattering and verticality
            geof = libply_c.compute_geof(xyz, target_fea, args.k_nn_geof).astype('float32')
            end = timer()
            times[0] = times[0] + end - start
            del target_fea

            write_features(file_name=fea_file, geof=geof, xyz=xyz, rgb=rgb,
                           graph_nn=graph_nn, labels=labels, intensity=intensity, nb_return=nb_return)

        # --compute the partition------
        sys.stdout.flush()
        if os.path.isfile(spg_file) and not args.overwrite:
            print("    reading the existing superpoint graph file...")
            graph_sp, components, in_component = read_spg(spg_file)
        else:
            print("    computing the superpoint graph...")
            # --- build the spg h5 file --
            start = timer()

            # choose here which features to use for the partition
            # In this examples, will use linearity, planarity, scattering, verticality, normalized intensity and number of returns.
            features = np.concatenate((geof, (intensity / max(intensity)), nb_return), axis=1).astype('float32')
            # geof[:, 3] = 2. * geof[:, 3]

            graph_nn["edge_weight"] = np.array(1. / (args.lambda_edge_weight + graph_nn["distances"] / np.mean(graph_nn["distances"])),
                                               dtype='float32')
            print("        minimal partition...")
            components, in_component = libcp.cutpursuit(features, graph_nn["source"], graph_nn["target"],
                                                        graph_nn["edge_weight"], args.reg_strength)
            components = np.array(components, dtype='object')
            end = timer()
            times[1] = times[1] + end - start
            print("        computation of the SPG...")
            start = timer()
            graph_sp = compute_sp_graph(xyz, args.d_se_max, in_component, components, labels, n_labels)
            end = timer()
            times[2] = times[2] + end - start
            write_spg(spg_file, graph_sp, components, in_component)

        print("Timer : %5.1f / %5.1f / %5.1f " % (times[0], times[1], times[2]))
