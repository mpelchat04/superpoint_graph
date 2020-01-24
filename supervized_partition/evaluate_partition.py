#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 11 10:12:49 2018

@author: landrieuloic
"""

"""
    Large-scale Point Cloud Semantic Segmentation with Superpoint Graphs
    http://arxiv.org/abs/1711.09869
    2017 Loic Landrieu, Martin Simonovsky
"""
import glob
import argparse
import sys
import h5py
sys.path.append("./learning")
from learning.metrics import *

parser = argparse.ArgumentParser(description='Evaluation function for S3DIS')

parser.add_argument('--odir', default='./results_partition/', help='Directory to store results')
parser.add_argument('--folder', default='', help='Directory to store results')
parser.add_argument('--dataset', default='s3dis', help='Directory to store results')
parser.add_argument('--cvfold', default='123456', help='which fold to consider')

args = parser.parse_args()
args.odir = args.odir + args.dataset + '/'


root = args.odir + args.folder + '/'

if args.dataset == 's3dis':
    fold_size = [44, 40, 23, 49, 68, 48]
    files = glob.glob(root + 'cv{}'.format(args.cvfold[0]) + '/res*.h5')
    n_classes = 13
elif args.dataset == 'vkitti':
    fold_size = [15, 15, 15, 15, 15, 15]
    files = glob.glob(root + '0{}'.format(args.cvfold[0]) + '/res*.h5')
    n_classes = 13

file_result_txt = open(args.odir + args.folder + '/results' + '.txt', "w")
file_result_txt.write("   N \t ASA \t BR \t BP\n")
    
    
C_classes = np.zeros((n_classes, n_classes))
C_BR = np.zeros((2, 2))
C_BP = np.zeros((2, 2))
N_sp = 0
N_pc = 0
    
for i_fold in range(len(args.cvfold)):
    fold = int(args.cvfold[i_fold])
    if args.dataset == 's3dis':
        base_name = root + 'cv{}'.format(fold) 
    elif args.dataset == 'vkitti':
        base_name = root + '0{}'.format(fold) 
        
    try:
        file_name = base_name + '/res.h5'
        res_file = h5py.File(file_name, 'r')
    except OSError:
        raise NameError('Cant find pretrained model %s' % file_name)
            
    c_classes = np.array(res_file["confusion_matrix_classes"])
    c_BP = np.array(res_file["confusion_matrix_BP"])
    c_BR = np.array(res_file["confusion_matrix_BR"])
    n_sp = np.array(res_file["n_clusters"])
    print(f"Fold {fold} : \t n_sp = {n_sp} \t ASA = {100 * c_classes.trace() / c_classes.sum():.2f}% "
          f"\t BR = {100 * c_BR[1,1] / (c_BR[1,1] + c_BR[1,0]):.2f}% \t BP = {100 * c_BP[1,1] / (c_BP[1,1] + c_BP[0,1]):.2f}%")
    C_classes += c_classes
    C_BR += c_BR
    C_BP += c_BP
    N_sp += n_sp * fold_size[i_fold]
    N_pc += fold_size[i_fold]
    
if N_sp > 0:
    print(f"\nOverall : \t n_sp = {N_sp/N_pc:.0f}  \t ASA = {100 * C_classes.trace() / C_classes.sum():.2f}% "
          f"\t BR = {100 * C_BR[1,1] / (C_BR[1,1] + C_BR[1,0]):.2f}% \t BP = {100 * C_BP[1,1] / (C_BP[1,1] + C_BP[0,1]):.2f}%\n")

    file_result_txt.write(f"{N_sp/N_pc:.0f} \t {100 * C_classes.trace() / C_classes.sum():.2f} "
                          f"\t {100 * C_BR[1,1] / (C_BR[1,1] + C_BR[1,0]):.2f}% \t {100 * C_BP[1,1] / (C_BP[1,1] + C_BP[0,1]):.2f}%\n")

file_result_txt.close()
