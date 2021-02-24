mkdir -p log
ps -ef|grep python|awk '{print $2}'|xargs kill -9

export MXNET_CPU_WORKER_NTHREADS=2
export MXNET_CPU_PRIORITY_NTHREADS=2
export MXNET_MP_WORKER_NTHREADS=2
export MXNET_KVSTORE_REDUCTION_NTHREADS=2
export OMP_NUM_THREADS=2
export PS_VERBOSE=1


export PSERVER_IP_LIST="127.0.0.1,127.0.0.1"
export TRAINER_PORTS="9092,9093"
export PSERVERS_NUM=2
export TRAINERS_NUM=2
eval $(./distributed_env.py)

export DMLC_ROLE=scheduler 
python -u train.py &> log/scheduler.log &

export DMLC_ROLE=server
for((i=0;i<$PSERVERS_NUM;i++))
do
    python -u train.py &> log/server.$i.log &
done

export DMLC_ROLE=worker
for((i=0;i<$TRAINERS_NUM;i++))
do
    python -u train.py &> log/worker.$i.log &
done
