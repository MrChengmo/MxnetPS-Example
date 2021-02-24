# Copyright (c) 2019 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
import sys
import six
import os
import copy
import logging
import numpy as np
import mxnet as mx
from mxnet import gluon
from mxnet.gluon.data import dataset

cont_min_ = [0, -3, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
cont_max_ = [20, 600, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
cont_diff_ = [20, 603, 100, 50, 64000, 500, 100, 50, 500, 10, 10, 10, 50]
hash_dim_ = 1000001
continuous_range_ = range(1, 14)
categorical_range_ = range(14, 40)

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Reader(dataset.Dataset):
    def set_file_list(self, file_list):
        self.all_data = []
        self.file_list = file_list
        self.example_nums = self.get_example_num(file_list)

    def get_example_num(self, file_list):
        count = 0
        for f in file_list:
            last_count = count
            for _, _ in enumerate(open(f, 'r')):
                count += 1
            logger.info("File: %s has %s examples" % (f, count - last_count))
        logger.info("Total example: %s" % count)
        return count

    def line_process(self, line):
        features = line.rstrip('\n').split('\t')
        dense_feature = []
        sparse_feature = []
        for idx in continuous_range_:
            if features[idx] == "":
                dense_feature.append(0.0)
            else:
                dense_feature.append(
                    (float(features[idx]) - cont_min_[idx - 1]) /
                    cont_diff_[idx - 1])
        for idx in categorical_range_:
            sparse_feature.append(hash(str(idx) + features[idx]) % hash_dim_)
        label = [int(features[0])]
        return label, sparse_feature, dense_feature

    def load_criteo_dataset(self, file_list):
        label, dense_feature = [], []
        sparse_feature = []
        for i in range(26):
            sparse_feature.append([])

        for file in file_list:
            with open(file, 'r') as f:
                for line in f:
                    input_data = self.line_process(line)
                    label.append(input_data[0])
                    dense_feature.append(input_data[-1])
                    for i in range(26):
                        sparse_feature[i].append(input_data[1][i])

        train_set = gluon.data.ArrayDataset(
            np.array(label),
            np.array(sparse_feature[0]).astype("float32"),
            np.array(sparse_feature[1]).astype("float32"),
            np.array(sparse_feature[2]).astype("float32"),
            np.array(sparse_feature[3]).astype("float32"),
            np.array(sparse_feature[4]).astype("float32"),
            np.array(sparse_feature[5]).astype("float32"),
            np.array(sparse_feature[6]).astype("float32"),
            np.array(sparse_feature[7]).astype("float32"),
            np.array(sparse_feature[8]).astype("float32"),
            np.array(sparse_feature[9]).astype("float32"),
            np.array(sparse_feature[10]).astype("float32"),
            np.array(sparse_feature[11]).astype("float32"),
            np.array(sparse_feature[12]).astype("float32"),
            np.array(sparse_feature[13]).astype("float32"),
            np.array(sparse_feature[14]).astype("float32"),
            np.array(sparse_feature[15]).astype("float32"),
            np.array(sparse_feature[16]).astype("float32"),
            np.array(sparse_feature[17]).astype("float32"),
            np.array(sparse_feature[18]).astype("float32"),
            np.array(sparse_feature[19]).astype("float32"),
            np.array(sparse_feature[20]).astype("float32"),
            np.array(sparse_feature[21]).astype("float32"),
            np.array(sparse_feature[22]).astype("float32"),
            np.array(sparse_feature[23]).astype("float32"),
            np.array(sparse_feature[24]).astype("float32"),
            np.array(sparse_feature[25]).astype("float32"),
            np.array(dense_feature).astype("float32")
        )
        return train_set

    def __len__(self):
        return self.example_nums

    def __getitem__(self, idx):
        return self.all_data[idx]
