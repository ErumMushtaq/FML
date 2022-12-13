import copy
import logging

import torch
import wandb
import numpy as np
import sys
from torch import nn
import os
import math
from sklearn.metrics import roc_auc_score

sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

from torch.optim import lr_scheduler
from time import time, sleep
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from fed_fets2021.dataset_creation_scripts.paths import *
from fed_fets2021.metric import softmax_dice2
try:
    from fedml_core.trainer.model_trainer import ModelTrainer
    from fedml_api.distributed.fedavg import Evaluation
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer
    # from FedML.fedml_api.distributed.fedavg import Evaluation

class CentralizedTrainer(object):
    r"""
    This class is used to train federated non-IID dataset in a centralized way
    """

    def __init__(self, dataset, model, device, args):
        logging.info("CentralizedTrainerFeTS init begin")
        self.device = device
        self.args = args
        self.model = model
        [train_data_num, test_data_num, train_data_global, test_data_global] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num * self.args.local_epoch
        self.test_data_num_in_total = test_data_num

        self.batch_dice = True #Taken from nnunet
        self.use_progress_bar = False
        # loss function
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        #optimizer and scheduler
        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30
        # self.initial_lr = 3e-4
        self.initial_lr = args.lr
        self.weight_decay = 3e-5
        self.args.num_batches_per_epoch = args.local_epoch * int(self.train_data_num_in_total/2)
        self.optimizer = torch.optim.Adam(self.model.parameters(), self.initial_lr, weight_decay=self.weight_decay,
                                          amsgrad=True)
        self.lr_scheduler = lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.2,
                                                           patience=self.lr_scheduler_patience,
                                                           verbose=True, threshold=self.lr_scheduler_eps,
                                                           threshold_mode="abs")
        self.model.to(self.device)
        #Evaluation Variables
        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []
        self.all_val_eval_metrics = []

        self.dice_kidney = 0
        self.dice_tumor = 0
        self.dice_tumor_kid = 0
        self.last_rounds_dice = 0
        self.this_rounds_dice = 0
        # self.train(self.train_global, self.de)

        # Few More variables
        ################# LEAVE THESE ALONE ################################################
        self.val_eval_criterion_MA = None
        self.train_loss_MA = None
        self.best_val_eval_criterion_MA = None
        self.best_MA_tr_loss_for_patience = None
        self.best_epoch_based_on_MA_tr_loss = None
        self.all_tr_losses = []
        self.all_val_losses = []
        self.all_val_losses_tr_mode = []
        self.all_val_eval_metrics = []  # does not have to be used

        ################# THESE DO NOT NECESSARILY NEED TO BE MODIFIED #####################
        self.patience = 50
        self.val_eval_criterion_alpha = 0.9  # alpha * old + (1-alpha) * new
        # if this is too low then the moving average will be too noisy and the training may terminate early. If it is
        # too high the training will take forever
        self.train_loss_MA_alpha = 0.93  # alpha * old + (1-alpha) * new
        self.train_loss_MA_eps = 5e-4  # new MA must be at least this much better (smaller)
        self.max_num_epochs = 1000
        self.num_batches_per_epoch = 240 #250
        self.num_val_batches_per_epoch = 25 #50
        self.also_val_in_tr_mode = False
        self.lr_threshold = 1e-6  # the network will not terminate training if the lr is still above this threshold
        self.last_rounds_dice = 0
        self.this_rounds_dice = 0
        self.training_loss_list = []
        self.whole_tumor_score_list = []
        self.core_region_score_list = []
        self.active_tumor_score_list = []
        self.actual_whole_tumor = []

    def train(self, train_data, device, args):
        self.model.to(device)
        if torch.cuda.is_available():
            with torch.cuda.device('cuda:7'):
                torch.cuda.empty_cache()

        train_losses_epoch = []

        # train one epoch

        self.model.train()
        logging.info("args.epochs:" + str(args.epochs))
        for epoch in range(args.epochs):
            logging.info("----------Training --------")
            logging.info("-----------EPOCH---"+str(epoch)+"-------")
            self.model.train()
            #Train:
            iteration_count = 0
            self.train_loss = 0.0
            running_loss = 0.0
            for sample in train_data:
                inputs = sample[0].to(device)
                labels = sample[1].to(device)
                # logging.info(torch.unique(labels))
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                iteration_count += 2
                if iteration_count % 100 == 0:
                    # args.is_debug == 1 and
                    # self.test(self.test_global, self.device, self.args, epoch)
                    logging.info("iterarion Count = %d, loss = %f," % (iteration_count, loss))
                if args.is_debug == 1 and  iteration_count == 2:
                    break
            self.train_loss = running_loss / iteration_count
            self.training_loss_list.append(self.train_loss)

            # self.lr_scheduler.step(epoch)
            logging.info(' Epoch ' + str(epoch) + ' Train_Loss '+str(self.train_loss))
            if epoch % self.args.frequency_of_the_test == 0:
                self.test(self.test_global, self.device, self.args, epoch)

            logging.info('---------- Training Loss Log ------------')
            logging.info(self.training_loss_list)
            # logging.info('----------- whole_tumor_score_list Log ----------')
            # logging.info(self.whole_tumor_score_list)
            # logging.info('----------- core_region_score_list Log -----------')
            # logging.info(self.core_region_score_list)
            # logging.info('----------- active_tumor_score_list Log -----------')
            # logging.info(self.active_tumor_score_list)


        # logging.info(' Client ID '+str(client_idx)+' training complete')

    def run_iteration(self, data_generator, device = None, do_backprop=True, run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)
            # data = to_cuda(data)
            # target = to_cuda(target)

        self.optimizer.zero_grad()
        output = self.model(data)
        del data
        l = self.loss(output, target)
        if do_backprop:
            l.backward()
            self.optimizer.step()

        if run_online_evaluation:
            self.run_online_evaluation(output, target)

        del target
        # logging.info('one iteration done ')
        return l.detach().cpu().numpy()

    def Dice_coef(self, output, target, eps=1e-5):  # dice score used for evaluation
        target = target.float()
        # logging.info("output size:" + str(output.shape))
        # logging.info("target size:" + str(target.shape))
        num = 2 * (output * target).sum()
        den = output.sum() + target.sum() + eps
        return num / den, den, num

    def evaluation(self, predictions, gt):
        # gt = gt.float()
        # predictions = predictions.float()
        # logging.info(torch.unique(predictions))
        # tk_pd = torch.gt(predictions, 0)
        # tk_gt = torch.gt(gt, 0)
        # logging.info("tk_pd size:" + str(tk_pd.shape))
        # logging.info("tk_gt size:" + str(tk_gt.shape))
        dice1, denom, num = self.Dice_coef((predictions == 1).float(), (gt == 1).float())
        dice2, denom, num = self.Dice_coef(torch.logical_or((predictions == 1), (predictions == 3 )).float(), torch.logical_or((gt == 1), (gt == 3)).float()) # 1 and 4
        dice3, denom, num = self.Dice_coef((predictions == 3).float(), (gt == 3).float()) # 4
        dice4, denom, num = self.Dice_coef((torch.gt(predictions, 0)).float(), (torch.gt(gt, 0)).float()) #1, 2, 4
        return dice1, dice2, dice3, dice4

    # def evaluation(self, predictions, gt):
    #     gt = gt.float()
    #     predictions = predictions.float()
    #     total, dice1, dice2, dice3 = softmax_dice2(gt, predictions)
    #     return dice1, dice2, dice3

    def run_online_evaluation(self, output, target):
        with torch.no_grad():
            num_classes = output.shape[1]
            output_softmax = softmax_helper(output)
            output_seg = output_softmax.argmax(1)

            # print(output_seg.shape)
            # print(target.shape)
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

        divisor = int(math.floor(self.test_data_num_in_total / 2))
        kidney_dice = self.dice_kidney.cpu().numpy()/divisor
        tumor_dice = self.dice_tumor.cpu().numpy()/divisor
        kidney_tumor_dice = self.dice_tumor_kid.cpu().numpy()/divisor

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        return global_dc_per_class[0], global_dc_per_class[1], kidney_dice, tumor_dice, kidney_tumor_dice

    def test(self, test_data, device, args, epoch):
        # Testing global model on the local data
        logging.info("----------testingg --------")
        model = self.model
        self.model.to(device)
        self.model.eval()

        if torch.cuda.is_available():
            with torch.cuda.device('cuda:7'):
                torch.cuda.empty_cache()

        iteration_count = 0
        whole_tumor_score = 0
        whole_tumor_score_list = []
        core_region_score = 0
        core_region_score_list = []
        active_tumor_score = 0
        active_tumor_score_list = []
        AWT_list = []
        with torch.no_grad():
            for (X, y) in test_data:
                iteration_count += 1
                if torch.cuda.is_available():
                    X = X.to(device)
                    y = y.to(device)
                y_pred = model(X).detach().cpu()
                preds_softmax = softmax_helper(y_pred)
                preds = preds_softmax.argmax(1)
                y = y.detach().cpu()

                iteration_count += 1
                # logging.info("preds size:" + str(preds.shape))
                # logging.info("y size:" + str(y.shape))
                dice1, dice2, dice3, dice4 = self.evaluation(preds, y)
                if iteration_count % self.args.frequency_of_the_test == 0:
                    logging.info("iterarion Count = %d, Acc = %f," % (iteration_count, ((dice1+dice2+dice3)/3)))
                whole_tumor_score_list.append(dice1)
                core_region_score_list.append(dice2)
                active_tumor_score_list.append(dice3)
                AWT_list.append(dice4)

                if args.is_debug == 1 and iteration_count == 1:
                    break
            whole_tumor_score = np.mean(whole_tumor_score_list)
            core_region_score = np.mean(core_region_score_list)
            active_tumor_score = np.mean(active_tumor_score_list)
            AWT_score = np.mean(AWT_list)
            self.whole_tumor_score_list.append(whole_tumor_score)
            self.core_region_score_list.append(core_region_score)
            self.active_tumor_score_list.append(active_tumor_score)
            self.actual_whole_tumor.append(AWT_score)

        self.this_rounds_dice = core_region_score + active_tumor_score + AWT_score
        if self.this_rounds_dice >= self.last_rounds_dice:
            self.last_rounds_dice = self.this_rounds_dice
            self.save_checkpoint('best_rounds_model', epoch)
            wandb.log({" Best Rounds Score (Tumor Core) ": core_region_score, "round ": epoch})
            # wandb.log({" Best Rounds Score (Label 1 - NCR) ": whole_tumor_score, "round ": epoch})
            # wandb.log({" Best Rounds Score (Label 2 - Edema) ": core_region_score, "round ": epoch})
            # wandb.log({" Best Rounds Score (ET) ": active_tumor_score, "round ": epoch})
            # wandb.log({" Best Rounds Score (Whole Tumor) ": AWT_score, "round ": epoch})
        print(' Server Eval: whole_tumor_score Acc ' + str(whole_tumor_score) + ' core_region_score Acc '+ str(core_region_score) + 'active_tumor_score Acc' + str(active_tumor_score) + ' Round '+str(epoch))
        # wandb.log({"Label 1": whole_tumor_score, "round ": epoch})
        # wandb.log({"(Label 2 - Edema)": core_region_score, "round ": epoch})
        # wandb.log({" Label 4 (ET) ": active_tumor_score, "round ": epoch})
        # wandb.log({" Whole tumor (WT) ": AWT_score, "round ": epoch})
        # wandb.log({" Train Loss": self.train_loss, "round ": epoch})
        # wandb.log({" lr ": self.optimizer.param_groups[0]['lr'], "round ": epoch})
        wandb.log({" Train Loss": self.train_loss, "round ": epoch})
        wandb.log({" lr ": self.optimizer.param_groups[0]['lr'], "round ": epoch})
        wandb.log({"Mean NCR_dice Dice": whole_tumor_score, "round ": epoch})
        wandb.log({"Mean TC_dice": core_region_score, "round ": epoch})
        wandb.log({"Mean Mean ET_dice": active_tumor_score, "round ": epoch})
        wandb.log({"Mean Mean WT_dice": AWT_score, "round ": epoch})

    def save_checkpoint(self, fname, round):
        start_time = time()
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        logging.info(" Saving Checkpoint...")
        save_this = {
            'epoch': round + 1,
            'state_dict': state_dict}
        torch.save(self.model.state_dict(),
                   base + 'CL_'+ '_RUN_ID_' + str(
                       self.args.run_id) + '_lr_' + str(self.args.lr) + '_round_' + str(
                       self.args.comm_round) + '_local_epoch_' + str(self.args.local_epoch) + '_Threshold_' + str(
                       self.args.threshold) + '_best_model.model')
        logging.info(' Model saved ')
        torch.save(save_this, base + 'CL_'+ '_RUN_ID_' + str(
            self.args.run_id) + '_lr_' + str(self.args.lr) + '_round_' + str(
            self.args.epochs) + '_local_epoch_' + str(self.args.local_epoch) + '_Threshold_' + str(
            self.args.threshold) + 'round_best_model.model')

    def manage_patience(self):
        # update patience
        continue_training = True
        if self.patience is not None:
            # if best_MA_tr_loss_for_patience and best_epoch_based_on_MA_tr_loss were not yet initialized,
            # initialize them
            if self.best_MA_tr_loss_for_patience is None:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA

            if self.best_epoch_based_on_MA_tr_loss is None:
                self.best_epoch_based_on_MA_tr_loss = self.epoch

            if self.best_val_eval_criterion_MA is None:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA

            # check if the current epoch is the best one according to moving average of validation criterion. If so
            # then save 'best' model
            # Do not use this for validation. This is intended for test set prediction only.
            # self.print_to_log_file("current best_val_eval_criterion_MA is %.4f0" % self.best_val_eval_criterion_MA)
            # self.print_to_log_file("current val_eval_criterion_MA is %.4f" % self.val_eval_criterion_MA)

            if self.val_eval_criterion_MA > self.best_val_eval_criterion_MA:
                self.best_val_eval_criterion_MA = self.val_eval_criterion_MA
                # self.print_to_log_file("saving best epoch checkpoint...")
                if self.save_best_checkpoint: self.save_checkpoint(join(self.output_folder, "model_best.model"))

            # Now see if the moving average of the train loss has improved. If yes then reset patience, else
            # increase patience
            if self.train_loss_MA + self.train_loss_MA_eps < self.best_MA_tr_loss_for_patience:
                self.best_MA_tr_loss_for_patience = self.train_loss_MA
                self.best_epoch_based_on_MA_tr_loss = self.epoch
                # self.print_to_log_file("New best epoch (train loss MA): %03.4f" % self.best_MA_tr_loss_for_patience)
            else:
                pass
                # self.print_to_log_file("No improvement: current train MA %03.4f, best: %03.4f, eps is %03.4f" %
                #                       (self.train_loss_MA, self.best_MA_tr_loss_for_patience, self.train_loss_MA_eps))

            # if patience has reached its maximum then finish training (provided lr is low enough)
            if self.epoch - self.best_epoch_based_on_MA_tr_loss > self.patience:
                if self.optimizer.param_groups[0]['lr'] > self.lr_threshold:
                    # self.print_to_log_file("My patience ended, but I believe I need more time (lr > 1e-6)")
                    self.best_epoch_based_on_MA_tr_loss = self.epoch - self.patience // 2
                else:
                    # self.print_to_log_file("My patience ended")
                    continue_training = False
            else:
                pass
                # self.print_to_log_file(
                #    "Patience: %d/%d" % (self.epoch - self.best_epoch_based_on_MA_tr_loss, self.patience))

        return continue_training

    def update_eval_criterion_MA(self):
        """
        If self.all_val_eval_metrics is unused (len=0) then we fall back to using -self.all_val_losses for the MA to determine early stopping
        (not a minimization, but a maximization of a metric and therefore the - in the latter case)
        :return:
        """
        if self.val_eval_criterion_MA is None:
            if len(self.all_val_eval_metrics) == 0:
                self.val_eval_criterion_MA = - self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.all_val_eval_metrics[-1]
        else:
            if len(self.all_val_eval_metrics) == 0:
                """
                We here use alpha * old - (1 - alpha) * new because new in this case is the vlaidation loss and lower
                is better, so we need to negate it.
                """
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA - (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_losses[-1]
            else:
                self.val_eval_criterion_MA = self.val_eval_criterion_alpha * self.val_eval_criterion_MA + (
                        1 - self.val_eval_criterion_alpha) * \
                                             self.all_val_eval_metrics[-1]

    def maybe_update_lr(self):
        # maybe update learning rate
        if self.lr_scheduler is not None:
            assert isinstance(self.lr_scheduler, (lr_scheduler.ReduceLROnPlateau, lr_scheduler._LRScheduler))

            if isinstance(self.lr_scheduler, lr_scheduler.ReduceLROnPlateau):
                # lr scheduler is updated with moving average val loss. should be more robust
                self.lr_scheduler.step(self.train_loss_MA)
            else:
                self.lr_scheduler.step(self.epoch + 1)

    def update_train_loss_MA(self):
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]