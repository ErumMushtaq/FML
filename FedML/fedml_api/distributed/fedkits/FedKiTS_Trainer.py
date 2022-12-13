import dill
import logging
import time
import numpy as np
import torch
import wandb
import math
from tqdm import trange
from time import time, sleep
import os
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.optim import lr_scheduler
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from fed_kits19.dataset_creation_scripts.paths import *
# from .utils.reproducibility import set_seed


class FedAvgTrainer_(ModelTrainer):
    def __init__(self, model, args=None):
        self.model = model
        self.args = args
        self.batch_dice = True #Taken from nnunet
        self.use_progress_bar = False

        # loss function
        self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

        #optimizer and scheduler
        self.lr_scheduler_eps = 1e-3
        self.lr_scheduler_patience = 30
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
        self.all_val_eval_metrics = []

        self.dice_kidney = 0
        self.dice_tumor = 0
        self.dice_tumor_kid = 0

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

        #Training
        self.use_progress_bar = True
        if 'nnunet_use_progress_bar' in os.environ.keys():
            self.use_progress_bar = bool(int(os.environ['nnunet_use_progress_bar']))

    def get_model_params(self):
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)

    def train(self, train_data, device, args, client_idx, N, round):

        self.model.to(device)
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        epoch_start_time = time()
        train_losses_epoch = []

        # train one epoch
        self.model.train()
        iteration_count = 0
        # logging.info('Training Starting')
        running_loss = 0.0
        iteration_count = 0
        self.train_loss = 0.0
        for epoch in range(args.epochs):
            logging.info("----------Training --------")
            logging.info("-------Epoch-"+str(epoch)+"-CI-"+str(client_idx)+"------")
            self.model.train()
            #Train:
            for sample in train_data:
                inputs = sample[0].to(device)
                labels = sample[1].to(device)
                # logging.info(inputs.shape)
                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                iteration_count += 2
                if iteration_count % self.args.frequency_of_the_test == 0:
                    logging.info("iterarion Count = %d, loss = %f," % (iteration_count, loss))
                if args.is_debug == 1 and iteration_count == 2:
                    break
        self.train_loss = running_loss / iteration_count
        # self.training_loss_list.append(self.train_loss)
        self.all_tr_losses.append(self.train_loss)
        self.update_train_loss_MA()
        # Take a step after local training
        # self.maybe_update_lr()
        train_loss = np.mean(train_losses_epoch)
        logging.info(' Client ID '+str(client_idx)+' training complete')
        logging.info('Epoch-{0} lr: {1}'.format(round, self.optimizer.param_groups[0]['lr']))
        return self.train_loss

    def update_train_loss_MA(self):
        if self.train_loss_MA is None:
            self.train_loss_MA = self.all_tr_losses[-1]
        else:
            self.train_loss_MA = self.train_loss_MA_alpha * self.train_loss_MA + (1 - self.train_loss_MA_alpha) * \
                                 self.all_tr_losses[-1]

    def run_iteration(self, data_generator, round, device = None, do_backprop=True,  run_online_evaluation=False):
        data_dict = next(data_generator)
        data = data_dict['data']
        target = data_dict['target']

        data = maybe_to_torch(data)
        target = maybe_to_torch(target)

        if torch.cuda.is_available():
            data = data.to(device)
            target = target.to(device)

        self.optimizer.zero_grad()
        output = self.model(data)
        del data
        l = self.loss(output, target)
        # logging.info(l)
        if do_backprop:
            l.backward()
            self.optimizer.step()
        # self.lr_scheduler.step(round + 1)

        del target
        # logging.info('one iteration done ')
        return l.detach().cpu().numpy()

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
        return tk_dice, tu_kid_dice, tu_dice

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

    def finish_online_evaluation(self, test_data_num):
        self.online_eval_tp = np.sum(self.online_eval_tp, 0)
        self.online_eval_fp = np.sum(self.online_eval_fp, 0)
        self.online_eval_fn = np.sum(self.online_eval_fn, 0)

        global_dc_per_class = [i for i in [2 * i / (2 * i + j + k) for i, j, k in
                                           zip(self.online_eval_tp, self.online_eval_fp, self.online_eval_fn)]
                               if not np.isnan(i)]
        self.all_val_eval_metrics.append(np.mean(global_dc_per_class))


        divisor = int(math.floor(test_data_num / 2))
        kidney_dice = self.dice_kidney.cpu().numpy()/divisor
        tumor_dice = self.dice_tumor.cpu().numpy()/divisor
        kidney_tumor_dice = self.dice_tumor_kid.cpu().numpy()/divisor

        self.online_eval_foreground_dc = []
        self.online_eval_tp = []
        self.online_eval_fp = []
        self.online_eval_fn = []

        #
        # self.update_eval_criterion_MA()
        # continue_training = self.manage_patience()

        return global_dc_per_class[0], global_dc_per_class[1], kidney_dice, tumor_dice, kidney_tumor_dice

    def test(self, test_data, test_data_num, device, args, round_idx=None):
        # Testing global model on the local data
        print("----------testing --------")
        model = self.model
        self.model.to(device)
        self.model.eval()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        iteration_count = 0
        kidney_dice_score = 0
        kidney_dice_score_list = []
        tumor_dice_score = 0
        tumor_dice_score_list = []
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
                dice1, dice2, dice3 = self.evaluation(preds, y)
                if iteration_count % self.args.frequency_of_the_test == 0:
                    logging.info("iterarion Count = %d, Acc = %f," % (iteration_count, ((dice2 + dice3) / 2)))
                kidney_dice_score_list.append(dice2)
                tumor_dice_score_list.append(dice3)

                if args.is_debug == 1 and iteration_count == 1:
                    break
            kidney_dice_score = np.mean(kidney_dice_score_list)
            tumor_dice_score = np.mean(tumor_dice_score_list)
            self.kidney_test_dice.append(kidney_dice_score)
            self.tumor_test_dice.append(tumor_dice_score)


        self.this_rounds_dice = kidney_dice_score + tumor_dice_score
        if self.this_rounds_dice >= self.last_rounds_dice:
            self.last_rounds_dice = self.this_rounds_dice
            self.save_checkpoint('best_rounds_model', round_idx)
            wandb.log({" Best Rounds Score (Tumor) ": tumor_dice_score, "round ": round_idx})
            wandb.log({" Best Rounds Score (Kidney) ": kidney_dice_score, "round ": round_idx})
        # self.maybe_update_lr()
        wandb.log({" lr ": self.optimizer.param_groups[0]['lr'], "round ": round})
        return (kidney_dice_score, tumor_dice_score, 0.0, 0.0, 0.0), 0.0


    # def maybe_update_lr(self, round):
    #     # maybe update learning rate
    #     if self.lr_scheduler is not None: # KiTS use moving average of training loss
    #         self.lr_scheduler.step(round + 1)
    #     logging.info("lr is now (scheduler) %s" % str(self.optimizer.param_groups[0]['lr']))

    def save_checkpoint(self, fname, round):
        start_time = time()
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        logging.info(" Saving Checkpoint...")
        save_this = {
            'epoch': round + 1,
            'state_dict': state_dict}
        torch.save(self.model.state_dict(), base + 'FL_' + '_RUN_ID_' +str(self.args.run_id) + '_lr_'+str(self.args.lr) + '_round_'+str(self.args.comm_round) + '_epoch_'+str(self.args.epochs)+'_best_model.model')
        logging.info(' Model saved ')
        torch.save(save_this, base + 'FL_' + '_RUN_ID_' +str(self.args.run_id) + '_lr_'+str(self.args.lr) + '_round_'+str(self.args.comm_round) + '_epoch_'+str(self.args.epochs)+'round_best_model.model')

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

        #
        # loss_list = []
        # acc_list = []

        # ## Add the evaluation function
        # iteration_count = 0
        # with torch.no_grad():
        #     for i, (images, labels) in enumerate(test_data):
        #         iteration_count += 1
        #         images = images.to(device)
        #         labels = labels.to(device)
        #
        #         # need to recheck
        #         # Forward pass
        #         outputs = model(images)
        #         loss = torch.nn.functional.cross_entropy(outputs, labels, reduce=None).detach()
        #         loss_list.append(loss.reshape(-1))
        #
        #         acc = (torch.argmax(outputs, dim=1) == labels).float().detach()
        #         acc_list.append(acc.reshape(-1))
        #
        #         losses = torch.cat(loss_list, dim=0).mean().cpu().data.numpy()
        #         accuracies = torch.cat(acc_list, dim=0).mean().cpu().data.numpy()

        # return (losses, accuracies), model

    def test_on_the_server(
            self, train_data_local_dict, test_data_local_dict, test_data_num, device, args=None, round_idx=None, mean_train_loss = 0.0
    ) -> bool:
        logging.info("----------test_on_the_server--------")
        # score, _ = self.test(test_data_local_dict[0], device, args, round_idx)
        self.dice_kidney = 0
        self.dice_tumor = 0
        self.dice_tumor_kid = 0
        score, val_loss = self.test(test_data_local_dict, test_data_num, device, args, round_idx)
        print(' Server Eval: Kidney Dice ' + str(score[0]) + ' Tumor Dice '+ str(score[1])+ ' Round '+str(round_idx))
        wandb.log({"Kidney Dice": score[0], "round ": round_idx})
        wandb.log({" Tumor Dice": score[1], "round ": round_idx})
        wandb.log({" Train Loss": mean_train_loss, "round ": round_idx})
        wandb.log({" lr ": self.optimizer.param_groups[0]['lr'], "round ": round_idx})

        self.training_loss_list.append(mean_train_loss)
        logging.info('---------- Training Loss Log ------------')
        logging.info(self.training_loss_list)
        logging.info('----------- Test Kidney Dice Log ----------')
        logging.info(self.kidney_test_dice)
        logging.info('----------- Test Tumor Dice Log -----------')
        logging.info(self.tumor_test_dice)

        # ckpt_dir = args.Experiment_buffer + '/ckpt/' + "server/"
        # direction_dir = args.Experiment_buffer + "directions/server"
        # if round_idx == 0:
        #     # Add deletion code as well if file exists but filled up from prev experiments
        #     if not os.path.exists(direction_dir):
        #         os.mkdir(direction_dir)
        #     if not os.path.exists(ckpt_dir):
        #         os.mkdir(ckpt_dir)
        # if "init" in args.save_strategy and round_idx == 0:  # check if round_idx required
        #     torch.save(self.model.state_dict(), f"{ckpt_dir}/init_model.pt", pickle_module=dill)
        # # elif "final" in args.save_strategy and round_idx == args.comm_round - 2:
        # #     torch.save(self.model.state_dict(), f"{ckpt_dir}/{round_idx + 1}_round_final_model.pt", pickle_module=dill)
        # # else:
        # #     torch.save(self.model.state_dict(), f"{ckpt_dir}/{round_idx + 1}_model.pt", pickle_module=dill)
        # # loss_list, acc_list = [], []
        # for client_idx in test_data_local_dict.keys():
        #     test_data = test_data_local_dict[client_idx]
        #     score, _ = self.test(test_data, device, args, round_idx)
        #     # model_list.append(model)
        #     loss_list.append(score[0])
        #     acc_list.append(score[1])
        #
        #     wandb.log({"Client {} Test/Loss".format(client_idx): score[0],
        #                "Client {} Test/Accuracy".format(client_idx): score[1]})
        #
        # # wandb.log({"Kidney Acc": global_dc_per_class[0], "epoch ": self.epoch})
        # # wandb.log({" Tumor Acc": global_dc_per_class[1], "epoch ": self.epoch})
        # avg_loss, avg_acc = np.mean(np.array(loss_list)), np.mean(np.array(acc_list))
        # logging.info("Test Loss = {}".format(avg_loss))
        # logging.info("Test Accuracy = {}".format(avg_acc))
        # wandb.log({"Test/Loss": avg_loss})
        # wandb.log({"Test/Accuracy": avg_acc})
        return True
