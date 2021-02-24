import math
import mxnet as mx
import numpy
from mxnet import gluon, init, symbol, np, npx
import mxnet.ndarray as nd
from mxnet.gluon import nn, loss
from sklearn.random_projection import johnson_lindenstrauss_min_dim

# npx.set_np()


class CtrDnn(nn.HybridBlock):
    def __init__(self, sparse_feature_number, sparse_feature_dim,
                 dense_feature_dim, num_field, layer_sizes, **kwargs):
        super(CtrDnn, self).__init__(**kwargs)
        self.sparse_feature_number = sparse_feature_number
        self.sparse_feature_dim = sparse_feature_dim

        sizes = [sparse_feature_dim * num_field +
                 dense_feature_dim] + layer_sizes

        self.embedding = nn.Embedding(
            sparse_feature_number, sparse_feature_dim, sparse_grad=True)

        self.dense1 = nn.Dense(in_units=sizes[0],
                               units=sizes[1],
                               activation='relu',
                               weight_initializer=mx.init.Normal(1.0 / math.sqrt(sizes[1])))

        self.dense2 = nn.Dense(in_units=sizes[1],
                               units=sizes[2],
                               activation='relu',
                               weight_initializer=mx.init.Normal(1.0 / math.sqrt(sizes[2])))

        self.dense3 = nn.Dense(in_units=sizes[2],
                               units=sizes[3],
                               activation='relu',
                               weight_initializer=mx.init.Normal(1.0 / math.sqrt(sizes[3])))

        self.dense4 = nn.Dense(in_units=layer_sizes[-1],
                               units=2,
                               weight_initializer=mx.init.Normal(1.0 / math.sqrt(sizes[-1])))

    def hybrid_forward(self, F, sparse_inputs, dense_inputs):
        sparse_embs = []
        for s_input in sparse_inputs:
            emb = self.embedding(s_input)
            sparse_embs.append(emb)

        for i in range(len(sparse_embs)):
            sparse_embs[i] = F.reshape(
                sparse_embs[i], (-1, self.sparse_feature_dim))

        dnn_input = F.concat(sparse_embs[0],
                             sparse_embs[1],
                             sparse_embs[2],
                             sparse_embs[3],
                             sparse_embs[4],
                             sparse_embs[5],
                             sparse_embs[6],
                             sparse_embs[7],
                             sparse_embs[8],
                             sparse_embs[9],
                             sparse_embs[10],
                             sparse_embs[11],
                             sparse_embs[12],
                             sparse_embs[13],
                             sparse_embs[14],
                             sparse_embs[15],
                             sparse_embs[16],
                             sparse_embs[17],
                             sparse_embs[18],
                             sparse_embs[19],
                             sparse_embs[20],
                             sparse_embs[21],
                             sparse_embs[22],
                             sparse_embs[23],
                             sparse_embs[24],
                             sparse_embs[25],
                             dense_inputs,
                             dim=1)
        layer1 = self.dense1(dnn_input)
        layer2 = self.dense2(layer1)
        layer3 = self.dense3(layer2)
        dnn_output = self.dense4(layer3)

        return dnn_output
