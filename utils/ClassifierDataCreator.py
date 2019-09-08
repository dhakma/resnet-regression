import argparse
import os
import shutil

data_dirs = ['path-multi-sketch-gen-5', 'area-multi-sketch-gen-5']
classification_dir = 'classification-data'
data_types = ['train', 'val', 'test']


def create_classification_data(base_dir):
    for orig_dir in data_dirs:
        for data_type in data_types:
            src_dir = os.path.join(base_dir, orig_dir, data_type)
            if not (os.path.exists(src_dir)):
                raise Exception(src_dir + ' does not exist')

    for data_type in data_types:
        for orig_dir in data_dirs:
            dest_dir = os.path.join(classification_dir, data_type, orig_dir)
            if os.path.exists(dest_dir):
                shutil.rmtree(dest_dir)
            os.makedirs(dest_dir)
            src_dir = os.path.join(base_dir, orig_dir, data_type)
            for entry in os.scandir(src_dir):
                if entry.name.endswith('.png'):
                    shutil.copy(entry.path, dest_dir)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='.')
    args = parser.parse_args()
    create_classification_data(args.data_dir)
