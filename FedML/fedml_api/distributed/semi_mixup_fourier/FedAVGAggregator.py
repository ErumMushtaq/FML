import copy
import logging
import random
import time

import numpy as np
import torch
import wandb
import math
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
import matplotlib.pyplot as plt
from .utils import transform_list_to_tensor


class FedAVGAggregator(object):

    def __init__(self, train_global, test_global, all_train_data_num,
                 train_data_local_dict, test_data_local_dict, train_data_local_num_dict, worker_num, device,
                 args, model_trainer, test_data_num, model):
        self.trainer = model_trainer
        self.test_data_num = test_data_num
        # self.model = model

        self.args = args
        self.train_global = train_global
        self.test_global = test_global
        self.val_global = self._generate_validation_set()
        self.all_train_data_num = all_train_data_num

        self.train_data_local_dict = train_data_local_dict
        self.test_data_local_dict = test_data_local_dict
        #
        # logging.info(self.test_data_local_dict)
        # exit()
        self.train_data_local_num_dict = train_data_local_num_dict

        self.worker_num = worker_num
        self.device = device
        self.model_dict = dict()
        self.sample_num_dict = dict()
        self.train_loss = dict()
        self.flag_client_model_uploaded_dict = dict()
        self.probabilities_dict = dict()

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
        self.all_val_eval_metrics = []

        self.dice_kidney = []
        self.dice_tumor = []
        self.dice_tumor_kid = []

        self.kidney_mean_val = dict()
        self.tumor_mean_val = dict()

        #predictions
        self.mean_predictions_kidney = dict()
        self.mean_predictions_tumor = dict()
        self.mean_predictions_all = dict()
        self.weight_ = dict()

        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False

    def get_global_model_params(self):
        return self.trainer.get_model_params()

    def exponential_rampup(self, round_idx):
        # ref: https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=9746957
        phase = 1 - round_idx/self.args.comm_round
        return 1 * float(np.exp(-5.0 * phase * phase))



    def set_global_model_params(self, model_parameters):
        # self.model.load_state_dict(model_parameters)
        self.trainer.set_model_params(model_parameters)

    def add_local_trained_result(self, index, model_params, sample_num, train_loss, round_idx):

        logging.info("add_model. index = %d" % index)
        self.model_dict[index] = model_params
        self.sample_num_dict[index] = sample_num
        if index in range(2, 6):
            self.weight_[index] = train_loss['weight']
        #     self.sample_num_dict[index] = sample_num * train_loss['weight']
        # else:
        #     self.sample_num_dict[index] = sample_num * 0.5
        # if index in range(2, 6):
        #     if self.args.weight_scheme == 'th_weight':
        #         self.sample_num_dict[index] = train_loss['pixels_ratio']
        #     elif self.args.weight_scheme == 'expo_weight':
        #         self.sample_num_dict[index] = sample_num*self.exponential_rampup(round_idx)
        #     elif self.args.weight_scheme == 'dr_weight':
        #         self.sample_num_dict[index] = 6
        # if index in range(0, 2) and self.args.weight_scheme == 'dr2_weight':
        #         self.sample_num_dict[index] = self.sample_num_dict[index]*3


        # if self.args.phase == 'student_training':
        #     self.mean_predictions_all[index] = train_loss['prediction_all']
        #     self.mean_predictions_kidney[index] = train_loss['prediction_kidney']
        #     self.mean_predictions_tumor[index] = train_loss['prediction_tumor']

        self.train_loss[index] = train_loss['train_loss']
        # self.kidney_mean_val[index] = train_loss['kidney_mean']
        # self.tumor_mean_val[index] = train_loss['tumor_mean']
        # self.probabilities_dict[index] = train_loss['softmax_probabilities']
        self.flag_client_model_uploaded_dict[index] = True
        logging.info(' N is '+str(self.sample_num_dict[index])+' CI '+str(index))
        if 'Kidney_Dice' in train_loss.keys():
            self.kidney_mean_val[index] = train_loss['Kidney_Dice']
            self.tumor_mean_val[index] = train_loss['Tumor_Dice']


    def check_whether_all_receive(self):
        logging.debug("worker_num = {}".format(self.worker_num))
        for idx in range(self.worker_num):
            if not self.flag_client_model_uploaded_dict[idx]:
                return False
        for idx in range(self.worker_num):
            self.flag_client_model_uploaded_dict[idx] = False
        return True

    # def dynamic_weight(self, Client_ID_dict):
    #     labeled_data_count = 0
    #     index = 0
    #     unlabed_CI_pool = []
    #     selected_CI_pool = []
    #     for CI in Client_ID_dict:
    #         if CI == 1:
    #             labeled_data_count = Client_ID_dict + self.sample_num_dict[index]
    #         else:
    #             unlabed_CI_pool.append(index)
    #         index += 1

    #     CI_found = 0
    #     selected_list = []
    #     N = labeled_data_count
    #     gap = labeled_data_count
    #     while CI_found == 0:
    #         unlabel_CI = random.choice(unlabed_CI_pool)
    #         if self.sample_num_dict[unlabel_CI] < gap:
    #             selected_list.append(unlabel_CI)
    #         elif self.sample_num_dict[unlabel_CI] == gap:
    #             selected_list.append(unlabel_CI)






    def aggregate(self):
        start_time = time.time()
        model_list = []
        training_num = 0

        for idx in range(self.worker_num):
            if self.args.is_mobile == 1:
                self.model_dict[idx] = transform_list_to_tensor(self.model_dict[idx])
            model_list.append((self.sample_num_dict[idx], self.model_dict[idx]))
            training_num += self.sample_num_dict[idx]

        logging.info("len of self.model_dict[idx] = " + str(len(self.model_dict)))

        # logging.info("################aggregate: %d" % len(model_list))
        (num0, averaged_params) = model_list[0]
        for k in averaged_params.keys():
            for i in range(0, len(model_list)):
                local_sample_number, local_model_params = model_list[i]
                w = local_sample_number / training_num
                if i == 0:
                    averaged_params[k] = local_model_params[k] * w
                else:
                    averaged_params[k] += local_model_params[k] * w

        # update the global model which is cached at the server side
        self.set_global_model_params(averaged_params)

        end_time = time.time()
        logging.info("aggregate time cost: %d" % (end_time - start_time))
        return averaged_params

    def client_sampling(self, round_idx, client_num_in_total, client_num_per_round):
        if client_num_in_total == client_num_per_round:
            client_indexes = [client_index for client_index in range(client_num_in_total)]
        else:
            num_clients = min(client_num_per_round, client_num_in_total)
            np.random.seed(round_idx)  # make sure for each comparison, we are selecting the same clients each round
            client_indexes = np.random.choice(range(client_num_in_total), num_clients, replace=False)
        logging.info("client_indexes = %s" % str(client_indexes))
        return client_indexes

    def _generate_validation_set(self, num_samples=10000):
        if self.args.dataset.startswith("stackoverflow"):
            test_data_num  = len(self.test_global.dataset)
            sample_indices = random.sample(range(test_data_num), min(num_samples, test_data_num))
            subset = torch.utils.data.Subset(self.test_global.dataset, sample_indices)
            sample_testset = torch.utils.data.DataLoader(subset, batch_size=self.args.batch_size)
            return sample_testset
        else:
            return self.test_global

    def Dice_coef(self, output, target, eps=1e-5):  # dice score used for evaluation
        target = target.float()
        num = 2 * (output * target).sum()
        den = output.sum() + target.sum() + eps
        return num / den, den, num

    def evaluation(self, predictions, gt):
        gt = gt.float()
        predictions = predictions.float()
        # Compute tumor+kidney Dice >0 (1+2)
        tk_pd = torch.gt(predictions, 0)
        tk_gt = torch.gt(gt, 0)
        tk_dice, denom, num = self.Dice_coef(tk_pd.float(), tk_gt.float())  # Composite
        tu_kid_dice, denom, num = self.Dice_coef((predictions == 1).float(), (gt == 1).float())
        tu_dice, denom, num = self.Dice_coef((predictions == 2).float(), (gt == 2).float())
        self.dice_kidney.append(tu_kid_dice)
        self.dice_tumor.append(tu_dice)
        self.dice_tumor_kid.append(tk_dice)


    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)

            self.evaluation(output_seg, target)
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

        # print(global_dc_per_class)
        # print(global_dc_per_class[0])
        # print(global_dc_per_class[1])
        # wandb.log({"Kidney Acc": global_dc_per_class[0], "epoch ": self.epoch})
        # wandb.log({" Tumor Acc": global_dc_per_class[1], "epoch ": self.epoch})

        # self.print_to_log_file("Average global foreground Dice:", [np.round(i, 4) for i in global_dc_per_class])
        # self.print_to_log_file("(interpret this as an estimate for the Dice of the different classes. This is not "
        #                        "exact.)")

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        return global_dc_per_class[0], global_dc_per_class[1]

    def test(self, test_data, device, args, round_idx=None):
        # Testing global model on the local data
        print("----------testingg --------")
        model = self.model
        self.model.to(device)
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        iteration_count = 0
        print(' Validation Data Length '+str(int(math.floor(self.test_data_num/2))))
        with torch.no_grad():
            for i in range(int(math.floor(self.test_data_num/2))):
                data_dict = next(test_data)
                data = data_dict['data']
                target = data_dict['target']
                data = maybe_to_torch(data)
                target = maybe_to_torch(target)

                if iteration_count % self.args.frequency_of_the_test == 0:
                    logging.info("Testing: iterarion Count = %d" % ( iteration_count))

                if torch.cuda.is_available():
                    data = data.to(device)
                    target = target.to(device)

                output = self.model(data)
                del data
                self.run_online_evaluation(output, target)
                del target

                iteration_count += 1
                # if iteration_count % self.args.frequency_of_the_test == 0:
                #     logging.info("iterarion Count = %d, loss = %f," % (iteration_count))


                if args.is_debug == 1 and iteration_count == 1:
                    break

                # self.run_iteration(test_data, do_backprop=False, run_online_evaluation=True)

        kidney_dice_score, tumor_dice_score = self.finish_online_evaluation()
        return (kidney_dice_score, tumor_dice_score), model

    def test_on_the_server(self, test_data_local_dict, device, args=None, round_idx=None):
        logging.info("----------test_on_the_server--------")
        score, _ = self.test(test_data_local_dict, device, args, round_idx)
        logging.info(' Server Eval: Kidney Acc ' + str(score[0]) + ' Tumor Acc '+ str(score[1])+ ' Round '+str(round_idx))
        wandb.log({"Kidney Acc": score[0], "round ": round_idx})
        wandb.log({" Tumor Acc": score[1], "round ": round_idx})

        wandb.log({"Kidney Acc (kits19)": self.dice_kidney, "round ": round_idx})
        wandb.log({"Tumor Acc (kits19)": self.dice_tumor, "round ": round_idx})
        wandb.log({"Kidney and Tumor Acc": self.dice_tumor_kid, "round ": round_idx})
        logging.info(' Server Eval: Kidney Acc ' + str(self.dice_kidney) + ' Tumor Acc ' + str(
            self.dice_tumor) + ' Round ' + str(round_idx))

        self.dice_kidney = []
        self.dice_tumor = []
        self.dice_tumor_kid = []

    def test_on_server_for_all_clients(self, round_idx):
        mean_train_loss = sum(self.train_loss.values()) / len(self.train_loss)
        wandb.log({"Mean Train Loss": mean_train_loss, "round ": round_idx})

        mean_sup_loss = (self.train_loss[0] + self.train_loss[1])/2
        wandb.log({" Sup Loss": mean_sup_loss, "round ": round_idx})

        for index in range(2,6):
            wandb.log({"Kidney Pseudo Label Dice Client ID"+str(index): self.kidney_mean_val[index], "round no. ": round_idx})
            wandb.log({" Tumor Pseudo Label Dice Client ID"+str(index): self.tumor_mean_val[index], "round no. ": round_idx})

        if len(self.train_loss) > 2:
            mean_unsup_loss = (self.train_loss[2] + self.train_loss[3] + self.train_loss[4]+ self.train_loss[5])/4
            wandb.log({" Unsup Loss": mean_unsup_loss, "round ": round_idx})

            wandb.log({" Weight w(t)":self.weight_[3],"round": round_idx})

        if self.trainer.test_on_the_server(self.train_data_local_dict, self.test_global, self.test_data_num, self.device, self.args, round_idx, mean_train_loss):
            return

        # # mean_train_loss = sum(self.train_loss.values()) / len(self.train_loss)
        # if self.args.phase == 'teacher_training':
        #     plt.figure()
        #     plt.hist(self.probabilities_dict[0], bins = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        #     plt.ylabel('Total Number of pixels ')
        #     plt.xlabel('Probabilities')
        #     plt.title("Histogram for C0")
        #     wandb.log({"Histogram for C0": [wandb.Image(plt)]})
        #     plt.close()

        #     plt.figure()
        #     plt.hist(self.probabilities_dict[1], bins = [0,0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1])
        #     plt.ylabel('Total Number of pixels ')
        #     plt.xlabel('Probabilities')
        #     plt.title("Histogram for C1")
        #     wandb.log({"Histogram for C1": [wandb.Image(plt)]})
        #     plt.close()


        # wandb.log({"Mean Kidney Prob Val C0 ": self.kidney_mean_val[0], "round ": round_idx})
        # wandb.log({"Mean Tumor Prob Val C0 ": self.tumor_mean_val[0], "round ": round_idx})

        # wandb.log({"Mean Kidney Prob Val C1 ": self.kidney_mean_val[1], "round ": round_idx})
        # wandb.log({"Mean Tumor Prob Val C1 ": self.tumor_mean_val[1], "round ": round_idx})

        # if self.args.phase == 'student_training':
        #     mean_prediction_all = sum(self.mean_predictions_all.values()) / len(self.mean_predictions_all)
        #     mean_prediction_kidney = sum(self.mean_predictions_kidney.values())/len(self.mean_predictions_kidney)
        #     mean_prediction_tumor = sum(self.mean_predictions_tumor.values())/len(self.mean_predictions_tumor)
        #     wandb.log({"Mean Tumor Prediction Dice ": mean_prediction_tumor, "round ": round_idx})
        #     wandb.log({"Mean Kidney Prediction Dice ": mean_prediction_kidney, "round ": round_idx})
        #     wandb.log({"Mean All Prediction Dice ": mean_prediction_all, "round ": round_idx})
        # logging.info(self.probabilities_dict[0])
        # wandb_table_prob_co = wandb.Table(columns=["Probabilities of C0"])
        # wandb_table_prob_co.add_data(str(self.probabilities_dict[0]))
        # wandb.log({" Probabilities of C0 ": wandb_table_prob_co})
        #
        # wandb_table_prob_c1 = wandb.Table(columns=["Probabilities of C1"])
        # wandb_table_prob_c1.add_data(str(self.probabilities_dict[1]))
        # wandb.log({" Probabilities of C1 ": wandb_table_prob_c1})


        # self.test_on_the_server(self.test_global, self.device, self.args, round_idx)
        # #     return
        #
        # if round_idx % self.args.frequency_of_the_test == 0 or round_idx == self.args.comm_round - 1:
        #     logging.info("################test_on_server_for_all_clients : {}".format(round_idx))
        #     train_num_samples = []
        #     train_tot_corrects = []
        #     train_losses = []
        #     for client_idx in range(self.args.client_num_in_total):
        #         # train data
        #         metrics = self.trainer.test(self.train_data_local_dict[client_idx], self.device, self.args)
        #         train_tot_correct, train_num_sample, train_loss = metrics['test_correct'], metrics['test_total'], metrics['test_loss']
        #         train_tot_corrects.append(copy.deepcopy(train_tot_correct))
        #         train_num_samples.append(copy.deepcopy(train_num_sample))
        #         train_losses.append(copy.deepcopy(train_loss))
        #
        #         """
        #         Note: CI environment is CPU-based computing.
        #         The training speed for RNN training is to slow in this setting, so we only test a client to make sure there is no programming error.
        #         """
        #         if self.args.ci == 1:
        #             break
        #
        #     # test on training dataset
        #     train_acc = sum(train_tot_corrects) / sum(train_num_samples)
        #     train_loss = sum(train_losses) / sum(train_num_samples)
        #     wandb.log({"Train/Acc": train_acc, "round": round_idx})
        #     wandb.log({"Train/Loss": train_loss, "round": round_idx})
        #     stats = {'training_acc': train_acc, 'training_loss': train_loss}
        #     logging.info(stats)
        #
        #     # test data
        #     test_num_samples = []
        #     test_tot_corrects = []
        #     test_losses = []
        #
        #     if round_idx == self.args.comm_round - 1:
        #         metrics = self.trainer.test(self.test_global, self.device, self.args)
        #     else:
        #         metrics = self.trainer.test(self.val_global, self.device, self.args)
        #
        #     test_tot_correct, test_num_sample, test_loss = metrics['test_correct'], metrics['test_total'], metrics[
        #         'test_loss']
        #     test_tot_corrects.append(copy.deepcopy(test_tot_correct))
        #     test_num_samples.append(copy.deepcopy(test_num_sample))
        #     test_losses.append(copy.deepcopy(test_loss))
        #
        #     # test on test dataset
        #     test_acc = sum(test_tot_corrects) / sum(test_num_samples)
        #     test_loss = sum(test_losses) / sum(test_num_samples)
        #     wandb.log({"Test/Acc": test_acc, "round": round_idx})
        #     wandb.log({"Test/Loss": test_loss, "round": round_idx})
        #     stats = {'test_acc': test_acc, 'test_loss': test_loss}
        #     logging.info(stats)
