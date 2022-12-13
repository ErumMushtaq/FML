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
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../")))
sys.path.insert(0, os.path.abspath(os.path.join(os.getcwd(), "../../../../FedML")))
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc

# from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from FedML.fedml_core.trainer.model_trainer import ModelTrainer
from nnunet.utilities.to_torch import maybe_to_torch, to_cuda
from torch.optim import lr_scheduler
from torch.nn.modules.loss import CrossEntropyLoss
from nnunet.training.loss_functions.dice_loss import DC_and_CE_loss
from nnunet.utilities.nd_softmax import softmax_helper
from nnunet.utilities.tensor_utilities import sum_tensor
from fed_kits19.loss import get_current_consistency_weight, update_ema_variables, DiceLoss
from nnunet.paths import *
import shutil
import numpy.ma as ma
from dataset_creation_scripts.nnunet_library.data_augmentations import fixmatch_transformations
# from .utils.reproducibility import set_seed


class FedAvgTrainer_(ModelTrainer):
    def __init__(self, model, ema_model, loss, ltr, utr, vtr, args=None):
        self.model = model #ema_model
        # self.ema_model = ema_model 
        # self.global_model = ema_model
        self.args = args
        self.batch_dice = True #Taken from nnunet
        self.use_progress_bar = False

        # loss function
        self.loss = loss
        self.weak_tr_transform, self.strong_tr_transform = fixmatch_transformations()
        # self.loss = DC_and_CE_loss({'batch_dice': self.batch_dice, 'smooth': 1e-5, 'do_bg': False}, {})

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

        self.weak_tr = ltr
        self.strong_tr = utr
        self.val_tr = vtr

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
        self.pixels_ratio = 0

        self.ce_loss = CrossEntropyLoss()
        self.dice_loss = DiceLoss(4)

        #Training
        self.use_progress_bar = True
        if 'nnunet_use_progress_bar' in os.environ.keys():
            self.use_progress_bar = bool(int(os.environ['nnunet_use_progress_bar']))

        self.training_loss_list = []
        self.kidney_test_dice = []
        self.tumor_test_dice = []
        self.probabilities_list = {}
        self.tumor_mean_val = 0.0
        self.kidney_mean_val = 0.0

        # prediction evaluation
        self.mean_kidney_predic = 0.0
        self.mean_tumor_predic = 0.0
        self.mean_all_predic = 0.0

    def get_model_params(self):
        logging.info(" Sup Model weights ")
        return self.model.cpu().state_dict()
        # return self.model.cpu().state_dict()

    def get_model_params_unsup(self):
        logging.info(" Unsup Model weights ")
        return self.model.cpu().state_dict()

    def set_model_params(self, model_parameters):
        logging.info("set_model_params")
        self.model.load_state_dict(model_parameters)
        # self.ema_model.load_state_dict(model_parameters)

    # unsupervised silo
    def unsuptrain(self, train_data, device, args, client_idx, N, round_, Client_lu_ID):
        # if round_  < 500:
        #     self.train_loss = 0.0
        #     dict = {'train_loss': self.train_loss, 'weight': 0.0}
        #     logging.info("----Passing Round Unsupervised---")
        #     return dict
        self.model.to(device)
        # self.ema_model.to(device)
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
            # for sample in train_data:
            for i_batch, sampled_batch in enumerate(train_data):
                # print(iteration_count)
                           
                # becuase weak TR does not have Numpy to Tensor transform
                volume_batch = maybe_to_torch(sampled_batch[0])
                label_batch = maybe_to_torch(sampled_batch[1])

                volume_batch = volume_batch.to(device)
                label_batch = label_batch.to(device)

                str_unlabeled_volume_batch = self.strong_tr_transform(volume_batch)
                if args.ssl_method =='nvidia':
                    weak_unlabeled_volume_batch = volume_batch
                else:
                    weak_unlabeled_volume_batch = self.weak_tr_transform(volume_batch)
                input_data = torch.cat([weak_unlabeled_volume_batch, str_unlabeled_volume_batch], dim=0)
                outputs = self.model(input_data)
                outputs_soft = torch.softmax(outputs, dim=1)
                outputs_soft_u_w, outputs_soft_u_s = outputs_soft.chunk(2)
                max_probs, batch_one_encod = torch.max(outputs_soft_u_w, dim=1, keepdim=True)
                mask = max_probs.ge(self.args.consistency_threshold).float()
                unsupervised_loss  = self.dice_loss(outputs_soft_u_s, batch_one_encod, mask = mask)
                consistency_weight = get_current_consistency_weight(iteration_count//150, self.args.consistency, self.args.consistency_rampup)
                loss = consistency_weight * unsupervised_loss

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                # update_ema_variables(self.model, self.ema_model, self.args.ema_decay, iteration_count)
                iteration_count += 1
                del volume_batch, label_batch, outputs, outputs_soft
                running_loss += loss.item() 
                
                if iteration_count % self.args.frequency_of_the_test == 0:
                    logging.info("iterarion Count = %d, loss = %f," % (iteration_count, loss))
                if self.args.is_debug == 1 and iteration_count == 2:
                    break
            self.train_loss = running_loss / iteration_count
            self.training_loss_list.append(self.train_loss)
        dict = {'train_loss': self.train_loss, 'weight': 1.0}
        # return train_loss
        return dict

    # supervised silo
    def suptrain(self, train_data, device, args, client_idx, N, round_, Client_lu_ID):
        self.model.to(device)
        train_losses_epoch = []

        # train one epoch
        self.model.train()
        iteration_count = 0
        for epoch in range(self.args.epochs):
            logging.info("----------Training --------")
            logging.info("-----------EPOCH---"+str(epoch)+"-------")
            # self.ema_model.train()
            #Train:
            
            self.train_loss = 0.0
            running_loss = 0.0
            # for sample in train_data:
            for i_batch, sampled_batch in enumerate(train_data):      
                # becuase weak TR does not have Numpy to Tensor transform
                volume_batch = maybe_to_torch(sampled_batch[0])
                label_batch = maybe_to_torch(sampled_batch[1])

                volume_batch = volume_batch.to(device)
                label_batch = label_batch.to(device)
                outputs = self.model(volume_batch)
                outputs_soft = torch.softmax(outputs, dim=1)
                # loss = (self.loss(outputs, label_batch))
                # print(label_batch)
                # print(label_batch.unsqueeze(1).long().shape)
                # print(outputs.shape)
                # exit()

                loss_ce = self.ce_loss(outputs, label_batch.squeeze(1).long())
                # loss_dice  = 0
                loss_dice = self.dice_loss(outputs_soft, label_batch)
                # loss_dice = self.dice_loss(outputs_soft[:1], label_batch[:1].unsqueeze(1))
                # loss = self.loss(outputs, label_batch)
                loss = (loss_dice + loss_ce)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
                iteration_count += 1

                running_loss += loss.item() 
                
                if iteration_count % self.args.frequency_of_the_test == 0:
                    logging.info("iterarion Count = %d, loss = %f," % (iteration_count, loss))
                if self.args.is_debug == 1 and iteration_count == 2:
                    break
            self.train_loss = running_loss / iteration_count
            self.training_loss_list.append(self.train_loss)
        dict = {'train_loss': self.train_loss, 'weight': 1.0}
        # return train_loss
        return dict

    def train(self, train_data, device, args, client_idx, N, round, Client_lu_ID):
        if self.args.apply_threshold  == 'True' and self.args.phase == 'student_training' and Client_lu_ID == 0:
            self.loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {}, ignore_label=255)
        else:
            self.loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {})

        # _ = train_data.next()
        self.model.to(device)
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        epoch_start_time = time()
        train_losses_epoch = []

        # train one epoch
        self.model.train()
        iteration_count = 0
        # logging.info('Training Starting')
        self.training_loss_list = []
        kidney_mean_prob = []
        tumor_mean_prob = []
        self.probabilities_list = []
        for epoch in range(args.epochs):
            logging.info("----------Training --------")
            logging.info("-----------EPOCH---" + str(epoch) + "-------")
            self.model.train()
            # Train:
            iteration_count = 0
            self.train_loss = 0.0
            running_loss = 0.0               
            for sample in train_data:
                # becuase weak TR does not have Numpy to Tensor transform
                # sample[0] = maybe_to_torch(sample[0])
                # sample[1] = maybe_to_torch(sample[1])
                # print(sample[0].shape)
                case_identifier = sample[2]
                data_dictt = {'data': sample[0].numpy(), 'seg': sample[1].numpy()}
                data_dict = self.weak_tr(**data_dictt)

                if Client_lu_ID == 0: #unlabelled
                    prediction_folder = os.path.join(base, 'Semi_FL_Teacher_Model_Predictions')
                    data_folder = os.path.join(prediction_folder, 'Run_ID_' + str(self.args.run_id))
                    client_idx_folder = os.path.join(data_folder, 'Client_idx_' + str(client_idx))
                    prediction_folder_dir = os.path.join(client_idx_folder,
                                                        'FL_' + '_RUN_ID_' + str(self.args.run_id) + '_lr_' + str(
                                                            self.args.lr) + '_local_epoch_' + str(
                                                            self.args.epochs) + '_round_' + str(
                                                            self.args.comm_round) + '_Threshold_' + str(
                                                            self.args.threshold))
                    mask_ = np.zeros(sample[1].shape, dtype=np.float32)
                    for i in range(sample[0].shape[0]):
                        data_one_case = np.load(os.path.join(prediction_folder_dir, "%s.npz" % case_identifier[i]))
                        # sample[0][i, :, :, :, :] = data_one_case['data']
                        # sample[1][i, :, :, :] = data_one_case['target']
                        # mask_[i, :, :, :] = data_one_case['mask']
                        sample[0][i, :, :, :, :] = maybe_to_torch(data_one_case['data'])
                        sample[1][i, :, :, :] = maybe_to_torch(data_one_case['target'])
                        mask_[i, :, :, :] = maybe_to_torch(data_one_case['mask'])
                    # Now apply Weak and augmentations
                    data_dictt = {'data': sample[0].numpy(), 'seg': sample[1].numpy()}
                    data_dictt = self.weak_tr(**data_dictt)
                    data_dict = self.strong_tr(**data_dictt)
                    
                data = data_dict['data']
                target = data_dict['seg']
                sample[0] = maybe_to_torch(data)
                sample[1] = maybe_to_torch(target)
                inputs = sample[0].to(device)
                labels = sample[1].to(device)

                outputs = self.model(inputs)
                loss = self.loss(outputs, labels)

                ### histograms
                if self.args.phase == 'teacher_training':
                    out = outputs.detach()
                    output_softmax = softmax_helper(out)
                    probabilities = np.max(output_softmax.cpu().numpy(), 1)
                    classes = np.argmax(output_softmax.cpu().numpy(), 1)
                    kidney_mask = np.zeros(classes.shape, dtype=np.float32)
                    tumor_mask = np.zeros(classes.shape, dtype=np.float32)
                    km_ = np.reshape(kidney_mask, (-1,))
                    tm_ = np.reshape(tumor_mask, (-1,))
                    probs_ = np.reshape(probabilities, (-1,))
                    cls_ = np.reshape(classes, (-1,))
                    km_[cls_ == 1] = 1
                    tm_[cls_ == 2] = 1
                    list1 = probs_.tolist()

                    self.probabilities_list.append(list1)
                    kidney_mean_prob.append(ma.masked_array(probs_, km_).mean())
                    tumor_mean_prob.append(ma.masked_array(probs_, tm_).mean())
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
            self.training_loss_list.append(self.train_loss)

        # self.lr_scheduler.step(round + 1)
        self.all_tr_losses.append(np.mean(self.training_loss_list))
        self.update_train_loss_MA()

        # Take a step after local training
        self.maybe_update_lr()
        logging.info(" Client l/unl "+str(Client_lu_ID)+ " iteration count "+str(iteration_count))
        train_loss = np.mean(self.training_loss_list)
        logging.info(' Client ID '+str(client_idx)+' training complete')
        if self.args.phase == 'teacher_training':
            self.kidney_mean_val = np.mean(kidney_mean_prob)
            self.tumor_mean_val = np.mean(tumor_mean_prob)
            flatten_list = sum(self.probabilities_list, [])
        else:
            flatten_list = []
        # print('Epoch-{0} lr: {1}'.format(round, self.optimizer.param_groups[0]['lr']))
        dict = {'train_loss': train_loss, 'kidney_mean': self.kidney_mean_val, 'tumor_mean': self.tumor_mean_val, 'softmax_probabilities':flatten_list, 'pixels_ratio':self.pixels_ratio, 'prediction_kidney':self.mean_kidney_predic, 'prediction_tumor':self.mean_tumor_predic, 'prediction_all':self.mean_all_predic}
        # return train_loss
        return dict


    def sigmoid_rampup(self, current, rampup_length):
        """Exponential rampup from https://arxiv.org/abs/1610.02242"""
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))

    def make_predictions(self, unlabeled_data, device, args, utrain_data_num, client_idx, round_idx):
        # Testing global model on the local data
        # print("---------- Load Best Teacher Model ---------")
        # self.load_teacher_model()

        model = self.model
        self.model.to(device)
        self.model.eval()
        loss = DC_and_CE_loss({'batch_dice': True, 'smooth': 1e-5, 'do_bg': False}, {}, ignore_label=255)

        logging.info("---------- Creating Predictions Path ---------Round_ID--"+str(round_idx))
        prediction_folder = os.path.join(base, 'Semi_FL_Teacher_Model_Predictions')
        maybe_mkdir_p(prediction_folder)
        data_folder = os.path.join(prediction_folder, 'Run_ID_'+str(self.args.run_id))
        maybe_mkdir_p(data_folder)
        client_idx_folder = os.path.join(data_folder, 'Client_idx_'+str(client_idx))
        maybe_mkdir_p(client_idx_folder)

        prediction_folder_dir = os.path.join(client_idx_folder, 'FL_' + '_RUN_ID_' + str(self.args.run_id) + '_lr_' + str(
                                                     self.args.lr) + '_local_epoch_' + str(
                                                     self.args.epochs) + '_round_' + str(
                                                     self.args.comm_round) + '_Threshold_' + str(self.args.threshold))

        # prediction_folder_dir = os.path.join(data_folder_dir, client_ID)

        maybe_mkdir_p(prediction_folder_dir)
        # maybe_mkdir_p(prediction_folder_dir)
        #
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

        shutil.rmtree(prediction_folder_dir)
        # logging.info(' Length of Prediction Folder '+str(len(os.listdir(prediction_folder_dir))))
        os.mkdir(prediction_folder_dir)
        iteration_count = 0
        probabilities_list = [] #for histogram
        prob_list_round = []
        self.tumor_mean_val = 0
        self.kidney_mean_val = 0.0
        pixels_ = 0
        self.pixels_ratio = 0.0

        mean_kidney_predic = []
        mean_tumor_predic = []
        mean_all_predic = []


        with torch.no_grad():
            logging.info(" Train Length " + str(utrain_data_num) + " Client ID " + str(client_idx))
            for (data, target, case_identifier) in unlabeled_data:
                data = maybe_to_torch(data)
                target = maybe_to_torch(target)
                
                data_dictt = {'data': data, 'seg': target}
                data_dict = self.val_tr(**data_dictt)

                iteration_count += 1
                if torch.cuda.is_available():
                    data = data_dict['data']
                    target = data_dict['target']
                    data = data.to(device)
                    target = target.to(device)

                output = self.model(data)
                # logging.info('Output shape '+str(output.shape))
                output_softmax = softmax_helper(output)
                # logging.info('Output softmax ' + str(output_softmax.shape))
                predictions = output_softmax.argmax(1, keepdim = True)

                tk_dice, tu_kid_dice, tu_dice = self.evaluation(predictions, target)
                mean_kidney_predic.append(tu_kid_dice.cpu().numpy().astype(np.float32))
                mean_all_predic.append(tk_dice.cpu().numpy().astype(np.float32))
                mean_tumor_predic.append(tu_dice.cpu().numpy().astype(np.float32))

                # logging.info('Predictions ' + str(predictions.shape))
                # Masking and thresholding
                probabilities, classes = torch.max(output_softmax, 1, keepdim = True)
                probs_ = torch.reshape(probabilities, (-1,))
                prob_list_round.append(probs_)
                # self.probabilities_list[iteration_count] = probs_
                
                #  val = loss(output, target)
                # probabilities = probabilities.cpu().numpy()
                # logging.info(predictions.shape)
                # logging.info(probabilities.shape)
                # logging.info(probabilities[1, 1, 1, 1:30])
                thresh = 0.0
                if self.args.apply_threshold == 'True':
                    if self.args.threshold_type == 'constant':
                        thresh = self.args.thresholded_value
                    else:
                        gap = self.args.max_thresholded_value - self.args.thresholded_value
                        thresh = self.args.thresholded_value+gap*self.sigmoid_rampup(round_idx, self.args.comm_round-1)
                    Mask = np.zeros(predictions.shape, dtype=np.float32)
                    predictions[probabilities < thresh] = 255
                    probabilities = probabilities.cpu().numpy()
                    Mask[probabilities > thresh] = 1
                    # logging.info(" Thresholded Value "+str(thresh))
                else:
                    # logging.info(" No Thresholded Value ")
                    Mask = np.ones(predictions.shape, dtype=np.float32)

                pixels_ = np.count_nonzero(Mask) + pixels_
                # logging.info(' Pixels of this round '+str(pixels_))
                self.tumor_mean_val = np.mean(Mask) + self.tumor_mean_val

                #save predictions (two separate)
                for i in range(data.shape[0]):
                    seg_ = predictions[i, :, :, :].cpu().numpy().astype(np.float32)
                    data_ = data[i, :, :, :].cpu().numpy().astype(np.float32)
                    # mask_ = Mask[i, :, :, :].cpu().numpy().astype(np.float32)
                    mask_ = Mask[i, :, :, :]
                    all_data = {'data': data_, 'seg': seg_, 'keys': case_identifier[i], 'mask': mask_}
                    np.savez_compressed(os.path.join(prediction_folder_dir, "%s.npz" % case_identifier[i]),
                                        data= data_,  target=seg_, mask= mask_)
                if args.is_debug == 1 and iteration_count == 1:
                    break

                del data, target, all_data

        logging.info(mean_kidney_predic)
        logging.info(mean_tumor_predic)
        self.tumor_mean_val = self.tumor_mean_val/iteration_count
        self.pixels_ratio = pixels_ /(1*128*128*128)
        logging.info(' Total number of images used for predictions are '+str(self.pixels_ratio))

        self.mean_kidney_predic = np.mean(mean_kidney_predic)
        self.mean_all_predic = np.mean(mean_all_predic)
        self.mean_tumor_predic = np.mean(mean_tumor_predic)
        # kidney_dice_score, tumor_dice_score, kidney_dice, tumor_dice, kidney_tumor_dice = self.finish_online_evaluation(50)
        # logging.info("----- Teacher Model Predictions Result -----")
        # logging.info(" Kidney Predictions Acc " + str(kidney_dice_score) + "prediction round " + str(round_idx))

    def run_online_evaluation__(self, output, target):
        with torch.no_grad():
            num_classes = 3
            # output_softmax = softmax_helper(output)
            # output_seg = output_softmax.argmax(1)
            output_seg = torch.squeeze(output)
            # print(num_classes)
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

    def test(self, test_data, test_data_num, device, args, round_idx=0):
        # test_data_local_dict, test_data_num, device, args, round_idx
        # Testing global model on the local data
        logging.info("----------testingg --------")
        # # # device = torch.cuda.device('cuda:1')
        # logging.info(test_data)
        # logging.info(test_data_num)
        # logging.info(device)

        model = self.model
        self.model.to(device)
        self.model.eval()

        # if torch.cuda.is_available():
        #     with torch.cuda.device('cuda:7'):
        #         torch.cuda.empty_cache()

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
                # logging.info(X.shape)
                iteration_count += 1
                # if torch.cuda.is_available():
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
            # self.whole_tumor_score_list.append(whole_tumor_score)
            # self.core_region_score_list.append(core_region_score)
            # self.active_tumor_score_list.append(active_tumor_score)
            # self.actual_whole_tumor.append(AWT_score)
        # if round_idx > 500:
        self.this_rounds_dice = core_region_score + active_tumor_score + AWT_score
        if self.this_rounds_dice >= self.last_rounds_dice:
            self.last_rounds_dice = self.this_rounds_dice
            self.save_checkpoint('best_rounds_model', round_idx)
            wandb.log({" Best Rounds Score (Tumor Core) ": core_region_score, "round ": round_idx})
        #     wandb.log({" Best Rounds Score (Label 1 - NCR) ": whole_tumor_score, "round ": epoch})
        #     wandb.log({" Best Rounds Score (Label 2 - Edema) ": core_region_score, "round ": epoch})
        #     wandb.log({" Best Rounds Score (ET) ": active_tumor_score, "round ": epoch})
        #     wandb.log({" Best Rounds Score (Whole Tumor) ": AWT_score, "round ": epoch})
        # print(' Server Eval: whole_tumor_score Acc ' + str(whole_tumor_score) + ' core_region_score Acc '+ str(core_region_score) + 'active_tumor_score Acc' + str(active_tumor_score) + ' Round '+str(epoch))
        # wandb.log({"Label 1": whole_tumor_score, "round ": epoch})
        # wandb.log({"(Label 2 - Edema)": core_region_score, "round ": epoch})
        # wandb.log({" Label 4 (ET) ": active_tumor_score, "round ": epoch})
        # wandb.log({" Whole tumor (WT) ": AWT_score, "round ": epoch})
        # wandb.log({" Train Loss": self.train_loss, "round ": epoch})
        # wandb.log({" lr ": self.optimizer.param_groups[0]['lr'], "round ": epoch})
        return (whole_tumor_score, core_region_score, active_tumor_score, AWT_score, 0.0), 0.0
    def save_checkpoint(self, fname, round_):
        start_time = time()
        state_dict = self.model.state_dict()
        for key in state_dict.keys():
            state_dict[key] = state_dict[key].cpu()

        logging.info(" Saving Checkpoint...")
        save_this = {
            'epoch': round_ + 1,
            'state_dict': state_dict}


        data_folder_dir = os.path.join(base, 'Best_Models')
        maybe_mkdir_p(data_folder_dir)
        model_save_folder = os.path.join(data_folder_dir, 'Run_ID_'+str(self.args.run_id))
        maybe_mkdir_p(model_save_folder)


        torch.save(self.model.state_dict(), model_save_folder + '/Semi_FL_Preprocess_'+str(self.args.phase)+ '_RUN_ID_' +str(self.args.run_id) + '_lr_'+str(self.args.lr) + '_round_'+str(self.args.comm_round) + '_epoch_'+str(self.args.epochs)+'_Threshold_'+str(self.args.threshold)+'_best_model.model')
        logging.info(' Model saved ')
        torch.save(save_this, model_save_folder + '/Semi_FL_Preprocess_' +str(self.args.phase)+ '_RUN_ID_' +str(self.args.run_id) + '_lr_'+str(self.args.lr) + '_round_'+str(self.args.comm_round) + '_epoch_'+str(self.args.epochs) +'_Threshold_'+str(self.args.threshold)+'round_best_model.model')
        # torch.save(save_this, 'best_model.model')
        # self.print_to_log_file("Done, Saving took %.2f seconds" % (time() - start_time))

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
            score, val_loss = self.test(test_data_local_dict, test_data_local_dict, device, args, round_idx)
            logging.info(' Server Eval: WT Dice ' + str(score[3]) + ' ET Dice '+ str(score[2])+ ' Round '+str(round_idx))
            wandb.log({"Mean NCR_dice Dice": score[0], "round ": round_idx})
            wandb.log({"Mean TC_dice": score[1], "round ": round_idx})
            wandb.log({"Mean Mean ET_dice": score[2], "round ": round_idx})
            wandb.log({"Mean Mean WT_dice": score[3], "round ": round_idx})
            wandb.log({" Train Loss": mean_train_loss, "round ": round_idx})
            wandb.log({" lr ": self.optimizer.param_groups[0]['lr'], "round ": round_idx})

            self.training_loss_list.append(mean_train_loss)
            return True
