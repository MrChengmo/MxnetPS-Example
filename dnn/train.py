from __future__ import print_function
import os
import sys
import logging
import random
import time
import numpy as np
from network import CtrDnn
from criteo_generator import Reader
from sklearn import metrics


from mxnet import autograd, gluon, kv, nd, np
import mxnet as mx

logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)


class Train(object):

    def run(self):
        # hyper parameters
        epochs = 1
        batch_size = 1000
        sparse_feature_number = 1000001
        sparse_feature_dim = 10
        dense_feature_dim = 13
        num_field = 26
        layer_sizes = [400, 400, 400]
        train_data_path = "./train_data"
        print_step = 5
        distributed_train = False
        cpu_num = int(os.getenv("CPU_NUM", 1))

        # create network
        ctx = mx.cpu()
        net = CtrDnn(sparse_feature_number, sparse_feature_dim,
                     dense_feature_dim, num_field, layer_sizes)
        net.initialize(ctx=ctx)
        # net.hybridize()

        self.loss = gluon.loss.SoftmaxCrossEntropyLoss()

        if distributed_train:
            self.store = kv.create('dist_async')
        else:
            self.store = kv.create('local')

        # Load the training data
        reader_start_time = time.time()

        file_list = self.get_file_list(train_data_path, distributed_train)
        reader = Reader()
        dataset = reader.load_criteo_dataset(file_list)
        train_data = gluon.data.DataLoader(
            dataset, batch_size, num_workers=cpu_num, last_batch="discard")
        reader_end_time = time.time()
        logger.info("Load Data in memory finish, using time: {}".format(
            reader_end_time - reader_start_time))

        if distributed_train:
            trainer = gluon.Trainer(net.collect_params(), 'adam', {
                'learning_rate': 0.0001, 'lazy_update': True}, kvstore=self.store, update_on_kvstore=True)
        else:
            trainer = gluon.Trainer(net.collect_params(), 'adam', {
                                    'learning_rate': 0.0001}, kvstore=self.store)

        for epoch in range(epochs):
            logger.info("Epoch {} training begin".format(epoch))
            epoch_start_time = time.time()

            batch_id = 1
            train_run_cost = 0.0
            total_examples = 0
            self.global_score = None
            self.global_label = None

            for batch in train_data:
                train_start = time.time()
                loss_value = self.train_batch(
                    batch, ctx, net, trainer)

                train_run_cost += (time.time() - train_start)
                total_examples += batch_size

                batch_id += 1
                if batch_id % print_step == 0:
                    metric_start = time.time()
                    fpr, tpr, _ = metrics.roc_curve(
                        list(self.global_lable.asnumpy()), list(self.global_score.asnumpy()))
                    auc_value = metrics.auc(fpr, tpr)
                    train_run_cost += (time.time() - metric_start)

                    metrics_string = "auc: {}, loss: {}".format(
                        auc_value, loss_value)
                    profiler_string = ""
                    profiler_string += "using_time: {} sec ".format(
                        train_run_cost)
                    profiler_string += "avg_batch_cost: {} sec, ".format(
                        format((train_run_cost) / print_step, '.5f'))
                    profiler_string += " ips: {} example/sec ".format(
                        format(total_examples / (train_run_cost), '.5f'))
                    logger.info("Epoch: {}, Batch: {}, {} {}".format(
                        epoch, batch_id, metrics_string, profiler_string))
                    train_run_cost = 0.0
                    total_examples = 0

            epoch_end_time = time.time()
            logger.info(
                "Epoch: {}, using time {} second,".format(
                    epoch, epoch_end_time - epoch_start_time))

    def calc_auc(self, label, output):
        output_exp = output.exp()
        paratition = output_exp.sum(axis=1, keepdims=True)
        score = output_exp / paratition
        score = nd.slice_axis(score, axis=1, begin=1, end=2)

        if self.global_score is None:
            # for first time
            self.global_score = score
            self.global_lable = label
        else:
            self.global_score = nd.concat(self.global_score, score, dim=0)
            self.global_lable = nd.concat(self.global_lable, label, dim=0)

    def forward_backward(self, network, label, sparse_input, dense_input):
        # Ask autograd to remember the forward pass
        with autograd.record():
            output = network(sparse_input, dense_input)
            losses = self.loss(output, label)
            self.calc_auc(label, output)

        for l in [losses]:
            l.backward()

        return np.mean(losses.as_np_ndarray())

    def train_batch(self, batch_list, context, network, gluon_trainer):
        label = batch_list[0]
        # label = gluon.utils.split_and_load(label, context)

        sparse_input = batch_list[1:-1]

        dense_input = batch_list[-1]

        # Run the forward and backward pass
        loss = self.forward_backward(network, label, sparse_input, dense_input)

        # Update the parameters
        this_batch_size = batch_list[0].shape[0]
        gluon_trainer.step(this_batch_size)

        return loss

    def get_example_num(self, file_list):
        count = 0
        for f in file_list:
            last_count = count
            for _, _ in enumerate(open(f, 'r')):
                count += 1
            logger.info("File: %s has %s examples" % (f, count - last_count))
        logger.info("Total example: %s" % count)
        return count

    def get_file_list(self, data_path, split_file_list=False):
        assert os.path.exists(data_path)
        file_list = [data_path + "/%s" % x for x in os.listdir(data_path)]
        file_list.sort()
        if split_file_list:
            file_list = self.get_file_shard(file_list)
        logger.info("File list: {}".format(file_list))
        self.get_example_num(file_list)
        return file_list

    def get_file_shard(self, files):
        if not isinstance(files, list):
            raise TypeError("files should be a list of file need to be read.")

        trainer_id = self.store.rank
        trainers = self.store.num_workers

        remainder = len(files) % trainers
        blocksize = int(len(files) / trainers)

        blocks = [blocksize] * trainers
        for i in range(remainder):
            blocks[i] += 1

        trainer_files = [[]] * trainers
        begin = 0
        for i in range(trainers):
            trainer_files[i] = files[begin:begin + blocks[i]]
            begin += blocks[i]

        return trainer_files[trainer_id]


if __name__ == "__main__":
    model = Train()
    model.run()
