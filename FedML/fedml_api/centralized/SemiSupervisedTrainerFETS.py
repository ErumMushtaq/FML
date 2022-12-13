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
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))

from torch.optim import lr_scheduler
from time import time, sleep
from torch.nn.modules.loss import CrossEntropyLoss
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss, DC_loss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from nnunet.paths import *
from collections import OrderedDict
# from FedML.fedml_api.centralized.losses import DiceLoss
from fed_kits19.loss import get_current_consistency_weight, update_ema_variables, DiceLoss
import random
import shutil
try:
    from fedml_core.trainer.model_trainer import ModelTrainer
    from fedml_api.distributed.fedavg import Evaluation
except ImportError:
    from FedML.fedml_core.trainer.model_trainer import ModelTrainer
    # from FedML.fedml_api.distributed.fedavg import Evaluation

class StudentTeacherTrainer(object):
    r"""
    This class is used to train federated non-IID dataset in a centralized way
    """

    def __init__(self, student_model, teacher_model, dataset, device, args):
        self.device = device
        self.args = args
        self.model = student_model
        self.ema_model = teacher_model
        [train_data_num, test_data_num, train_data_global, test_data_global] = dataset
        self.train_global = train_data_global
        self.test_global = test_data_global
        self.train_data_num_in_total = train_data_num * self.args.local_epoch
        self.test_data_num_in_total = test_data_num

        self.batch_dice = True #Taken from nnunet
        self.use_progress_bar = False

        # loss function
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})
        self.unsup_losss = DC_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

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
        self.round_counter = 0

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
        self.kidney_test_dice = []
        self.tumor_test_dice = []

        self.unsup_loss = 0.0
        self.sup_loss = 0.0
        self.weight_loss = 0.0
        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(4)

        self.whole_tumor_score_list = []
        self.core_region_score_list = []
        self.active_tumor_score_list = []
        self.actual_whole_tumor = []
        

    def train(self):

        self.model.to(self.device)
        self.ema_model.to(self.device)
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()


        train_losses_epoch = []

        # train one epoch
        self.model.train()
        iteration_count = 0
        for epoch in range(self.args.epochs):
            logging.info("----------Training --------")
            logging.info("-----------EPOCH---"+str(epoch)+"-------")
            self.model.train()
            #Train:
           
            self.train_loss = 0.0
            running_loss = 0.0
            unsup_loss = 0.0
            sup_loss = 0.0
            weight_loss = 0.0
            counter = 0
            # for sample in train_data:
            for i_batch, sampled_batch in enumerate(self.train_global):
                # print(iteration_count)                          
                # becuase weak TR does not have Numpy to Tensor transform
                volume_batch = maybe_to_torch(sampled_batch[0])
                label_batch = maybe_to_torch(sampled_batch[1])

                # print(volume_batch.shape)
                # print(label_batch.shape)

                volume_batch = volume_batch.to(self.device)
                label_batch = label_batch.to(self.device)

                unlabeled_volume_batch = volume_batch[1:]
                labeled_volume_batch = volume_batch[:1]

                # ICT mix factors
                ict_mix_factors = np.random.beta(
                    self.args.ict_alpha, self.args.ict_alpha, size=(2//2, 1, 1, 1, 1)) # batch size two, and one image is being combined with the other
                ict_mix_factors = torch.tensor(ict_mix_factors, dtype=torch.float).to(self.device)
                unlabeled_volume_batch_0 = unlabeled_volume_batch[0:1, ...]
                unlabeled_volume_batch_1 = unlabeled_volume_batch[1:, ...]

                # Mix images


                batch_ux_mixed = unlabeled_volume_batch_0 * \
                (1.0 - ict_mix_factors) + \
                unlabeled_volume_batch_1 * ict_mix_factors
                input_volume_batch = torch.cat([labeled_volume_batch, batch_ux_mixed], dim=0)

                outputs = self.model(input_volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                # logging.info(unlabeled_volume_batch_0.shape)
                # logging.info(ict_mix_factors.shape)
                # logging.info(outputs.shape)
                # Original consistency Loss
                with torch.no_grad():
                    ema_output_ux0 = torch.softmax(
                        self.ema_model(unlabeled_volume_batch_0), dim=1)
                    ema_output_ux1 = torch.softmax(
                        self.ema_model(unlabeled_volume_batch_1), dim=1)
                    batch_pred_mixed = ema_output_ux0 * \
                        (1.0 - ict_mix_factors) + ema_output_ux1 * ict_mix_factors
                # print(outputs[:1].shape)
                # print(label_batch[:1].shape)

                # Previous implementation
                # loss_ce = self.ce_loss(outputs[:1],
                #               label_batch[:1][:].squeeze(1).long())
                # # logging.info(outputs_soft[:1].shape)
                # # logging.info(label_batch[:1].shape)
                # loss_dice = self.dice_loss(outputs_soft[:1], label_batch[:1])
                # # batch_pred_mixed_one_encod.unsqueeze(1)
                # # loss_dice = self.dice_loss(outputs_soft[:1], label_batch[:1].unsqueeze(1))
                # supervised_loss = 0.5 * (loss_dice + loss_ce) # already has the ce loss 



                # # supervised_loss = 0.5 * (self.loss(outputs[:1], label_batch[:1]))
                # consistency_weight = get_current_consistency_weight(iteration_count//150, self.args.consistency, self.args.consistency_rampup)
                # batch_pred_mixed_one_encod = torch.argmax(batch_pred_mixed, dim=1)
                # # print(outputs_soft[:1].shape)
                # # print(batch_pred_mixed_one_encod.unsqueeze(1).shape)
                # unsup_loss_ce = self.ce_loss(outputs[1:],
                #               batch_pred_mixed_one_encod.long())
                # # consistency_loss = self.dice_loss(outputs_soft[1:], batch_pred_mixed_one_encod.unsqueeze(1)) 
                # consistency_loss = self.dice_loss(outputs_soft[1:], batch_pred_mixed_one_encod.unsqueeze(1)) + unsup_loss_ce
                # loss = supervised_loss + consistency_weight * (consistency_loss)

                # Updated Loss
                #outputs, labels
                batch_pred_mixed_one_encod = torch.argmax(batch_pred_mixed, dim=1)
                consistency_weight = get_current_consistency_weight(iteration_count//150, self.args.consistency, self.args.consistency_rampup)
                supervised_loss = self.loss(outputs[:1], label_batch[:1])
                consistency_loss = self.unsup_losss(outputs[1:], batch_pred_mixed_one_encod.unsqueeze(1))
                loss = supervised_loss + consistency_weight * (consistency_loss)


                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                update_ema_variables(self.model, self.ema_model, self.args.ema_decay, iteration_count)
                iteration_count += 1
                counter += 1
                running_loss += loss.item() 
                unsup_loss += consistency_loss.item()
                sup_loss += supervised_loss.item()
                weight_loss += consistency_weight
                
                if iteration_count % self.args.frequency_of_the_test == 0:
                    logging.info("iterarion Count = %d, loss = %f," % (iteration_count, loss))
                if self.args.is_debug == 1 and iteration_count == 2:
                    break
                wandb.log({" consistency weight": consistency_weight, " iteration_count ": iteration_count})
            self.train_loss = running_loss / counter
            self.unsup_loss = unsup_loss / counter
            self.sup_loss = sup_loss / counter
            self.weight_loss = weight_loss / counter
            self.training_loss_list.append(self.train_loss)

            # self.lr_scheduler.step(epoch)
            logging.info(' Epoch ' + str(epoch) + ' Train_Loss '+str(self.train_loss))
            if epoch % self.args.frequency_of_the_test == 0:
                self.test(self.test_global, self.device, self.args, epoch)

            # logging.info('---------- Training Loss Log ------------')
            # logging.info(self.training_loss_list)
            # logging.info('----------- Test Kidney Dice Log ----------')
            # logging.info(self.kidney_test_dice)
            # logging.info('----------- Test Tumor Dice Log -----------')
            # logging.info(self.tumor_test_dice)


    def Dice_coef(self, output, target, eps=1e-5):  # dice score used for evaluation
        target = target.float()
        num = 2 * (output * target).sum()
        den = output.sum() + target.sum() + eps
        return num / den, den, num

    def evaluation(self, predictions, gt):
        dice1, denom, num = self.Dice_coef((predictions == 1).float(), (gt == 1).float())
        dice2, denom, num = self.Dice_coef(torch.logical_or((predictions == 1), (predictions == 3 )).float(), torch.logical_or((gt == 1), (gt == 3)).float()) # 1 and 4
        dice3, denom, num = self.Dice_coef((predictions == 3).float(), (gt == 3).float()) # 4
        dice4, denom, num = self.Dice_coef((torch.gt(predictions, 0)).float(), (torch.gt(gt, 0)).float()) #1, 2, 4
        return dice1, dice2, dice3, dice4

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
            # wandb.log({" Best Rounds Score (Label 1 - NCR) ": whole_tumor_score, "round ": epoch})
            wandb.log({" Best Rounds Score (Tumor Core) ": core_region_score, "round ": epoch})
            # wandb.log({" Best Rounds Score (ET) ": active_tumor_score, "round ": epoch})
            # wandb.log({" Best Rounds Score (Whole Tumor) ": AWT_score, "round ": epoch})
        print(' Server Eval: whole_tumor_score Acc ' + str(whole_tumor_score) + ' core_region_score Acc '+ str(core_region_score) + 'active_tumor_score Acc' + str(active_tumor_score) + ' Round '+str(epoch))
        # wandb.log({"Label 1": whole_tumor_score, "round ": epoch})
        # wandb.log({"(Label 2 - Edema)": core_region_score, "round ": epoch})
        # wandb.log({" Label 4 (ET) ": active_tumor_score, "round ": epoch})
        # wandb.log({" Whole tumor (WT) ": AWT_score, "round ": epoch})
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
                   base + 'SemiCL_'+ '_RUN_ID_' + str(
                       self.args.run_id) + '_lr_' + str(self.args.lr) + '_round_' + str(
                       self.args.comm_round) + '_local_epoch_' + str(self.args.local_epoch) + '_Threshold_' + str(
                       self.args.threshold) + '_best_model.model')
        logging.info(' Model saved ')
        torch.save(save_this, base + 'SemiCL_'+ '_RUN_ID_' + str(
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

        # return (kidney_dice_score, tumor_dice_score), model
