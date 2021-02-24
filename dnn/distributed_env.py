#!/usr/bin/env python2
import os
try:  # py3
    from shlex import quote
except ImportError:  # py2
    from pipes import quote


def set_distributed_env():
    server_ip = os.getenv("PSERVER_IP_LIST")
    assert server_ip != None
    server_ip_list = server_ip.split(",")
    import socket
    addr = socket.getaddrinfo(server_ip_list[0], None)
    ip = addr[0][4][0]
    print("export DMLC_PS_ROOT_URI={}".format(quote(ip)))

    port_array = os.getenv("TRAINER_PORTS")
    assert port_array != None
    port_list = port_array.split(",")
    print("export DMLC_PS_ROOT_PORT={}".format(quote(port_list[0])))

    pservers_num = os.getenv("PSERVERS_NUM")
    assert pservers_num != None
    print("export DMLC_NUM_SERVER={}".format(quote(pservers_num)))

    trainers_num = os.getenv("TRAINERS_NUM")
    assert trainers_num != None
    print("export DMLC_NUM_WORKER={}".format(quote(trainers_num)))


if __name__ == "__main__":
    set_distributed_env()
