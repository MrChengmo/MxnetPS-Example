# Train with single machine

```bash
# set distributed_train = False in train.py at #34
python -u train.py
```

# Local Cluster Train

```bash
# set distributed_train = True in train.py at #34
bash local_cluster.sh
```

Running log  will save at `./log`