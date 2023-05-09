import numpy as np
import argparse
import os

from script_manager.func.add_needed_args import smart_parse_args


parser = argparse.ArgumentParser(description='confusion matrix analysis')
# dycs parameters
parser.add_argument('--confusion_matrix_file', default=None,
                    type=str, help='file where confusion matrix is stored')
parser.add_argument('--class-map', default=None,
                    type=str, help='class_map to provide mnemonic names')

args = smart_parse_args(parser)

def get_class_map_dict(class_map_file):
    label2wordnet = {}
    with open(class_map_file, "r") as f:
        lines = f.readlines()

    for i, l in enumerate(lines):
        if l[0] in ['n']:
            label2wordnet[i] = l.strip()
        else:
            pass

    total_classes = len(lines)
    return label2wordnet, total_classes

if __name__ == '__main__':

    label2wordnet, total_classes = get_class_map_dict(args.class_map)
    # C = np.load(args.confusion_matrix_file)
    # distance_mat = C.max()[0] * np.ones_like(C, dtype=np.float32) - C
    # K-means clustering of rows goes here
    n_clusters = 5

    file_path_base = args.confusion_matrix_file.split('.')[0]

    class_map_folders = file_path_base + '_decomposition'
    os.makedirs(class_map_folders, exist_ok=True)

    for c in range(n_clusters):
        cluster_classes = list(range(200*c, 200*(c+1)))
        cluster_classes = sorted(cluster_classes)

        cluster_file_path = os.path.join(
            class_map_folders, f'class_map{c}.txt')
        with open(cluster_file_path, "w") as f:
            for c in range(total_classes):
                if c in cluster_classes:
                    f.write(f'{label2wordnet[i]}')
                else:
                    f.write(f'{c}')
                if c != total_classes - 1:
                    f.write('\n')
