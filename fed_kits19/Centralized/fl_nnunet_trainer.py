import dill
import logging
import time
from time import time, sleep
import numpy as np
import torch
import wandb
from tqdm import trange
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.optim import lr_scheduler
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor

# from .utils.reproducibility import set_seed


class FedAvgTrainer_(ModelTrainer):
    def __init__(self, model, args=None):
        self.model = model
        self.args = args
        self.batch_dice = True #Taken from nnunet

        # loss function
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        #optimizer and scheduler
        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30
        # self.initial_lr = 3e-4
        self.initial_lr = args.lr
        self.weight_decay = 3e-5
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                          amsgrad=True)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                           patience=self.lr_scheduler_patience,
                                                           verbose=True, threshold=self.lr_scheduler_eps,
                                                           threshold_mode="abs")

        #Evaluation Variables
        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        #Training
        self.use_progress_bar = True
        if 'nnunet_use_progress_bar' in os.environ.keys():
            self.use_progress_bar = bool(int(os.environ['nnunet_use_progress_bar']))

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args):
        _ = train_data.next()
        self.model.to(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_start_time = time()
        train_losses_epoch = []

        # train one epoch
        self.model.train()
        iteration_count = 0
        running_loss = 0.0
        for epoch in range(args.epochs):
            #Train:
            iteration_count = 0
            for sample in train_data:
                inputs = sample[0].to(device)
                labels = sample[1].to(device)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                iteration_count += 2
            epoch_loss = running_loss / iteration_count


            if args.use_progress_bar:
                with trange(args.num_batches_per_epoch) as tbar:
                    for b in tbar:
                        tbar.set_description("Epoch {}/{}".format(self.epoch+1, self.max_num_epochs))
                        l = self.run_iteration(train_data, True)
                        tbar.set_postfix(loss=l)
                        train_losses_epoch.append(l)
            else:
                for _ in range(self.num_batches_per_epoch):
                    l = self.run_iteration(train_data, True)
                    train_losses_epoch.append(l)

        # logging.info(" Client ID " + str(client_idx) + " round Idx " + str(round_idx))
        # return max_test_score, best_model_params

    def run_iteration(self, data_generator, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = data.to(self.device)
            target = target.to(self.device)
            # data = to_cuda(data)
            # target = to_cuda(target)

        self.optimizer.zero_grad()
        output = self.network(data)
        del data
        l = self.loss(output, target)
        if do_backprop:
            l.backward()
            self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target

        return l.detach().cpu().numpy()

    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)
            target = target[:, 0]
            axes = tuple(range(1, len(target.shape)))
            tp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fp_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            fn_hard = torch.zeros((target.shape[0], num_classes - 1)).to(output_seg.device.index)
            for c in range(1, num_classes):
                tp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target == c).float(), axes=axes)
                fp_hard[:, c - 1] = sum_tensor((output_seg == c).float() * (target != c).float(), axes=axes)
                fn_hard[:, c - 1] = sum_tensor((output_seg != c).float() * (target == c).float(), axes=axes)

            tp_hard = tp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fp_hard = fp_hard.sum(0, keepdim=False).detach().cpu().numpy()
            fn_hard = fn_hard.sum(0, keepdim=False).detach().cpu().numpy()

            # print(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            # exit()
            self.online_eval_foreground_dc.append(list((2 * tp_hard) / (2 * tp_hard + fp_hard + fn_hard + 1e-8)))
            self.online_eval_tp.append(list(tp_hard))
            self.online_eval_fp.append(list(fp_hard))
            self.online_eval_fn.append(list(fn_hard))

    def finish_online_evaluation(self):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))

        print(global_dc_per_class)
        print(global_dc_per_class[0])
        print(global_dc_per_class[1])
        wandb.log({"Kidney Acc": global_dc_per_class[0], "epoch ": self.epoch})
        wandb.log({" Tumor Acc": global_dc_per_class[1], "epoch ": self.epoch})

        self.print_to_log_file("Average global foreground Dice:", [np.round(i, 4) for i in global_dc_per_class])
        self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
                               "exact.)")

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

    def test(self, test_data, device, args, round_idx=None):
        # Testing global model on the local data
        logging.info("----------test--------")
        model = self.model
        model.to(device)
        model.eval()

        loss_list = []
        acc_list = []

        ## Add the evaluation function
        iteration_count = 0
        with torch.no_grad():
            for i, (images, labels) in enumerate(test_data):
                iteration_count += 1
                images = images.to(device)
                labels = labels.to(device)

                # need to recheck
                # Forward pass
                outputs = model(images)
                loss = torch.nn.functional.cross_entropy(outputs, labels, reduce=None).detach()
                loss_list.append(loss.reshape(-1))

                acc = (torch.argmax(outputs, dim=1) == labels).float().detach()
                acc_list.append(acc.reshape(-1))

                losses = torch.cat(loss_list, dim=0).mean().cpu().data.numpy()
                accuracies = torch.cat(acc_list, dim=0).mean().cpu().data.numpy()

        return (losses, accuracies), model

    def test_on_the_server(
            self, train_data_local_dict, test_data_local_dict, device, args=None, round_idx=None
    ) -> bool:
        logging.info("----------test_on_the_server--------")

        ckpt_dir = args.Experiment_buffer + '/ckpt/' + "server/"
        direction_dir = args.Experiment_buffer + "directions/server"
        if round_idx == 0:
            # Add deletion code as well if file exists but filled up from prev experiments
            if not os.path.exists(direction_dir):
                os.mkdir(direction_dir)
            if not os.path.exists(ckpt_dir):
                os.mkdir(ckpt_dir)
        if "init" in args.save_strategy and round_idx == 0:  # check if round_idx required
            torch.save(self.model.state_dict(), f"{ckpt_dir}/init_model.pt", pickle_module=dill)
        # elif "final" in args.save_strategy and round_idx == args.comm_round - 2:
        #     torch.save(self.model.state_dict(), f"{ckpt_dir}/{round_idx + 1}_round_final_model.pt", pickle_module=dill)
        else:
            torch.save(self.model.state_dict(), f"{ckpt_dir}/{round_idx + 1}_model.pt", pickle_module=dill)
        loss_list, acc_list = [], []
        for client_idx in test_data_local_dict.keys():
            test_data = test_data_local_dict[client_idx]
            score, _ = self.test(test_data, device, args, round_idx)
            # model_list.append(model)
            loss_list.append(score[0])
            acc_list.append(score[1])

            wandb.log({"Client {} Test/Loss".format(client_idx): score[0],
                       "Client {} Test/Accuracy".format(client_idx): score[1]})
        avg_loss, avg_acc = np.mean(np.array(loss_list)), np.mean(np.array(acc_list))
        logging.info("Test Loss = {}".format(avg_loss))
        logging.info("Test Accuracy = {}".format(avg_acc))
        wandb.log({"Test/Loss": avg_loss})
        wandb.log({"Test/Accuracy": avg_acc})
        return True
