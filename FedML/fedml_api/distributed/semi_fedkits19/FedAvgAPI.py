from mpi4py import MPI

from .FedAVGAggregator import FedAVGAggregator
from .FedAVGTrainer import FedAVGTrainer
from .FedAvgClientManager import FedAVGClientManager
from .FedAvgServerManager import FedAVGServerManager
from .FedKiTS_Trainer import FedAvgTrainer_
# from ...standalone.fedavg.my_model_trainer_classification import MyModelTrainer as MyModelTrainerCLS
# from ...standalone.fedavg.my_model_trainer_nwp import MyModelTrainer as MyModelTrainerNWP
# from ...standalone.fedavg.my_model_trainer_tag_prediction import MyModelTrainer as MyModelTrainerTAG


def FedML_init():
    comm = MPI.COMM_WORLD
    process_id = comm.Get_rank()
    worker_number = comm.Get_size()
    return comm, process_id, worker_number


def FedML_FedAvg_distributed(process_id, worker_number, device, comm, model, train_data_num, train_data_global,
    test_data_global, train_data_local_num_dict, train_data_local_dict, test_data_local_dict, args, test_data_num, ltr, utr, vtr, Client_ID_dict, loss):
    # client_index = process_id - 1
    model_trainer = FedAvgTrainer_(model, loss, ltr, utr, vtr, args)
    if process_id == 0:
        init_server(args, device, comm, process_id, worker_number,  model, train_data_num, train_data_global, test_data_global, train_data_local_dict, test_data_local_dict,
        train_data_local_num_dict, model_trainer, test_data_num)
    else:
        init_client(args, device, comm, process_id, worker_number, model, train_data_num, train_data_local_num_dict,
        train_data_local_dict, test_data_local_dict, Client_ID_dict, model_trainer)


def init_server(args, device, comm, rank, size, model, train_data_num, train_data_global, test_data_global, train_data_local_dict,
    test_data_local_dict, train_data_local_num_dict, model_trainer, test_data_num):
    # aggregator
    worker_num = size - 1
    aggregator = FedAVGAggregator(train_data_global, test_data_global,train_data_num,
        train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device, args,
        model_trainer, test_data_num, model)

    # start the distributed training
    backend = args.backend
    server_manager = FedAVGServerManager(args, aggregator, comm, rank, size, backend)
    server_manager.send_init_msg()
    server_manager.run()


def init_client(
    args,
    device,
    comm,
    process_id,
    size,
    model,
    train_data_num,
    train_data_local_num_dict,
    train_data_local_dict,
    test_data_local_dict,
    Client_ID_dict,
    model_trainer=None
):
    client_index = process_id - 1
    model_trainer.set_id(client_index)
    backend = args.backend
    trainer = FedAVGTrainer(
        client_index,
        train_data_local_dict,
        train_data_local_num_dict,
        test_data_local_dict,
        train_data_num,
        device,
        args,
        model_trainer,
        Client_ID_dict,
    )
    client_manager = FedAVGClientManager(args, trainer, comm, process_id, size, backend)
    client_manager.run()
