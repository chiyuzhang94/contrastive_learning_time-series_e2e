from data.data_loader import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom, Dataset_Pred
from exp.exp_basic import Exp_Basic
from models.model import Informer, InformerStack
from models.tcn import TCN
from models.lstm import LSTM
from models.tcn_moco import TCN_MoCo
from models.dtcn_moco import DTCN_MoCo
from models.cost_e2e import COST_E2E
from models.dtcn import DTCN
from models.losses import hierarchical_contrastive_loss

from utils.tools import EarlyStopping, adjust_learning_rate, adjust_learning_rate_cos
from utils.metrics import metric

import numpy as np

import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import DataLoader

import os
import time, json

import warnings
warnings.filterwarnings('ignore')

class Exp_Informer(Exp_Basic):
    def __init__(self, args):
        super(Exp_Informer, self).__init__(args)
    
    def _build_model(self):
        model_dict = {
            'informer':Informer,
            'informerstack':InformerStack,
        }

        if self.args.model=='informer-encoder':
            model = TCN(input_size = self.args.enc_in,
                         represent_size = self.args.d_model, 
                         output_size = self.args.c_out, 
                         e_layer = self.args.e_layers, 
                         pred_leng = self.args.pred_len,
                         mask_rate = self.args.mask_rate,
                         freq = self.args.freq,
                         model_name = 'informer',
                         hidden_size = 64, 
                         contrastive_loss=self.args.loss_lambda,
                         dropout=self.args.dropout
                         )


        elif self.args.model=='informer-moco':
            model = TCN_MoCo(input_size = self.args.enc_in,
                             hidden_size = self.args.d_model, 
                             output_size = self.args.c_out, 
                             e_layer = self.args.e_layers, 
                             pred_leng = self.args.pred_len,
                             freq = self.args.freq,
                             kernel_size = self.args.kernel_size,
                             model_name = "informer",
                             dropout=self.args.dropout,
                             K=512,
                             T=1.00,
                             l2norm=self.args.l2norm,
                             mare=self.args.mare,
                             average_pool = self.args.moco_average_pool,
                             data_aug = self.args.data_aug,
                             time_feature_embed = self.args.time_feature_embed,
                             embed = self.args.embed,
                        )

        elif self.args.model == "informer-moco-hcl":
            model = TCN_MoCo(input_size = self.args.enc_in,
                             hidden_size = self.args.d_model, 
                             output_size = self.args.c_out, 
                             e_layer = self.args.e_layers, 
                             pred_leng = self.args.pred_len,
                             freq = self.args.freq,
                             model_name="informer", 
                             kernel_size = self.args.kernel_size,
                             dropout=self.args.dropout,
                             K=512,
                             T=1.00,
                             l2norm=self.args.l2norm,
                             mare=self.args.mare,
                             average_pool = self.args.moco_average_pool,
                             data_aug = self.args.data_aug,
                             tempral_cl = True,
                             mask_rate = self.args.mask_rate,
                             moco_cl_weight = 0.5,
                             time_feature_embed = self.args.time_feature_embed,
                             embed = self.args.embed
                        )

        elif self.args.model=='lstm':
            model = TCN(input_size = self.args.enc_in,
                         represent_size = self.args.d_model, 
                         output_size = self.args.c_out, 
                         e_layer = self.args.e_layers, 
                         pred_leng = self.args.pred_len,
                         mask_rate = self.args.mask_rate,
                         freq = self.args.freq,
                         model_name = 'lstm',
                         hidden_size = 64, 
                         contrastive_loss=self.args.loss_lambda,
                         dropout=self.args.dropout
                         )

        elif self.args.model == "lstm-moco":
            model = TCN_MoCo(input_size = self.args.enc_in,
                             hidden_size = self.args.d_model, 
                             output_size = self.args.c_out, 
                             e_layer = self.args.e_layers, 
                             pred_leng = self.args.pred_len,
                             freq = self.args.freq,
                             kernel_size = self.args.kernel_size,
                             model_name = "lstm",
                             dropout=self.args.dropout,
                             K=512,
                             T=1.00,
                             l2norm=self.args.l2norm,
                             mare=self.args.mare,
                             average_pool = self.args.moco_average_pool,
                             data_aug = self.args.data_aug,
                             time_feature_embed = self.args.time_feature_embed,
                             embed = self.args.embed,
                        )

        elif self.args.model == "lstm-moco-hcl":
            model = TCN_MoCo(input_size = self.args.enc_in,
                             hidden_size = self.args.d_model, 
                             output_size = self.args.c_out, 
                             e_layer = self.args.e_layers, 
                             pred_leng = self.args.pred_len,
                             freq = self.args.freq,
                             model_name="lstm", 
                             kernel_size = self.args.kernel_size,
                             dropout=self.args.dropout,
                             K=512,
                             T=1.00,
                             l2norm=self.args.l2norm,
                             mare=self.args.mare,
                             average_pool = self.args.moco_average_pool,
                             data_aug = self.args.data_aug,
                             tempral_cl = True,
                             mask_rate = self.args.mask_rate,
                             moco_cl_weight = 0.5,
                             time_feature_embed = self.args.time_feature_embed,
                             embed = self.args.embed
                        )

        elif self.args.model=='tcn':
            model = TCN(input_size = self.args.enc_in,
                         represent_size = self.args.d_model, 
                         output_size = self.args.c_out, 
                         e_layer = self.args.e_layers, 
                         pred_leng = self.args.pred_len,
                         mask_rate = self.args.mask_rate,
                         freq = self.args.freq,
                         hidden_size = 64,
                         model_name = 'tcn',
                         kernel_size = self.args.kernel_size,
                         contrastive_loss=self.args.loss_lambda,
                         dropout=self.args.dropout
                         )

        elif self.args.model == "tcn-moco":
            model = TCN_MoCo(input_size = self.args.enc_in,
                             hidden_size = self.args.d_model, 
                             output_size = self.args.c_out, 
                             e_layer = self.args.e_layers, 
                             pred_leng = self.args.pred_len,
                             freq = self.args.freq,
                             kernel_size = self.args.kernel_size,
                             dropout=self.args.dropout,
                             K=512,
                             T=1.00,
                             l2norm=self.args.l2norm,
                             mare=self.args.mare,
                             average_pool = self.args.moco_average_pool,
                             data_aug = self.args.data_aug,
                             time_feature_embed = self.args.time_feature_embed,
                             embed = self.args.embed,
                        )

        elif self.args.model == "tcn-moco-hcl":
            model = TCN_MoCo(input_size = self.args.enc_in,
                             hidden_size = self.args.d_model, 
                             output_size = self.args.c_out, 
                             e_layer = self.args.e_layers, 
                             pred_leng = self.args.pred_len,
                             freq = self.args.freq,
                             model_name="tcn", 
                             kernel_size = self.args.kernel_size,
                             dropout=self.args.dropout,
                             K=512,
                             T=1.00,
                             l2norm=self.args.l2norm,
                             mare=self.args.mare,
                             average_pool = self.args.moco_average_pool,
                             data_aug = self.args.data_aug,
                             tempral_cl = True,
                             mask_rate = self.args.mask_rate,
                             moco_cl_weight = 0.5,
                             time_feature_embed = self.args.time_feature_embed,
                             embed = self.args.embed
                        )


        elif self.args.model=='dtcn':
            model = DTCN(input_size = self.args.enc_in,
                         represent_size = self.args.d_model, 
                         output_size = self.args.c_out, 
                         e_layer = self.args.e_layers, 
                         pred_leng = self.args.pred_len,
                         mask_rate = self.args.mask_rate,
                         freq = self.args.freq,
                         hidden_size=64,
                         kernel_size = self.args.kernel_size,
                         contrastive_loss=self.args.loss_lambda,
                         dropout=self.args.dropout
                         )

        elif self.args.model == "dtcn-moco-hcl":
            model = TCN_MoCo(input_size = self.args.enc_in,
                             hidden_size = self.args.d_model, 
                             output_size = self.args.c_out, 
                             e_layer = self.args.e_layers, 
                             pred_leng = self.args.pred_len,
                             freq = self.args.freq,
                             model_name="dtcn", 
                             kernel_size = self.args.kernel_size,
                             dropout=self.args.dropout,
                             K=512,
                             T=1.00,
                             l2norm=self.args.l2norm,
                             mare=self.args.mare,
                             average_pool = self.args.moco_average_pool,
                             data_aug = self.args.data_aug,
                             tempral_cl = True,
                             mask_rate = self.args.mask_rate,
                             moco_cl_weight = 0.5,
                             time_feature_embed = self.args.time_feature_embed,
                             embed = self.args.embed
                        )

        elif self.args.model == "dtcn-moco":
            model = DTCN_MoCo(input_size = self.args.enc_in,
                             hidden_size = self.args.d_model, 
                             output_size = self.args.c_out, 
                             e_layer = self.args.e_layers, 
                             pred_leng = self.args.pred_len,
                             freq = self.args.freq,
                             kernel_size = self.args.kernel_size,
                             dropout=self.args.dropout,
                             K=512,
                             T=1.00,
                             l2norm=self.args.l2norm,
                             average_pool = self.args.moco_average_pool,
                             data_aug = self.args.data_aug,
                             mare=self.args.mare,
                             time_feature_embed = self.args.time_feature_embed,
                             embed = self.args.embed
                        )

        elif self.args.model == "cost-e2e":
            model = COST_E2E(input_size = self.args.enc_in,
                             hidden_size = self.args.d_model, 
                             output_size = self.args.c_out, 
                             e_layer = self.args.e_layers, 
                             pred_leng = self.args.pred_len,
                             input_length= self.args.seq_len,
                             dropout=self.args.dropout
                        )
            
        else:
            print("Wrong model name")
                        
        if self.args.use_multi_gpu and self.args.use_gpu:
            model = nn.DataParallel(model, device_ids=self.args.device_ids)

        pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

        print("[INFO] Number of parameters: ", pytorch_total_params)
        print(model)
        
        return model

    def _get_data(self, flag):
        args = self.args

        data_dict = {
            'ETTh1':Dataset_ETT_hour,
            'ETTh2':Dataset_ETT_hour,
            'ETTm1':Dataset_ETT_minute,
            'ETTm2':Dataset_ETT_minute,
            'WTH':Dataset_Custom,
            'ECL':Dataset_Custom,
            'Solar':Dataset_Custom,
            'custom':Dataset_Custom,
        }
        Data = data_dict[self.args.data]
        timeenc = 0 if args.embed!='timeF' else 1

        if flag == 'test' or flag == 'val':
            shuffle_flag = False; drop_last = True; batch_size = args.batch_size; freq=args.freq
        elif flag=='pred':
            shuffle_flag = False; drop_last = False; batch_size = 1; freq=args.detail_freq
            Data = Dataset_Pred
        else:
            shuffle_flag = True; drop_last = True; batch_size = args.batch_size; freq=args.freq
        data_set = Data(
            root_path=args.root_path,
            data_path=args.data_path,
            flag=flag,
            size=[args.seq_len, args.label_len, args.pred_len],
            features=args.features,
            target=args.target,
            inverse=args.inverse,
            timeenc=timeenc,
            freq=freq,
            cols=args.cols
        )
        print(flag, len(data_set))
        data_loader = DataLoader(
            data_set,
            batch_size=batch_size,
            shuffle=shuffle_flag,
            num_workers=args.num_workers,
            drop_last=drop_last)

        return data_set, data_loader

    def _select_optimizer(self):
        model_optim = optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return model_optim
    
    def _select_criterion(self):
        criterion =  nn.MSELoss()
        return criterion

    def vali(self, vali_data, vali_loader, criterion):
        self.model.eval()
        total_loss = []
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(vali_loader):
            pred, true, _, _ = self._process_one_batch(
                vali_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            loss = criterion(pred.detach().cpu(), true.detach().cpu())
            total_loss.append(loss)
        total_loss = np.average(total_loss)
        self.model.train()
        return total_loss

    def train(self, setting):
        train_data, train_loader = self._get_data(flag = 'train')
        vali_data, vali_loader = self._get_data(flag = 'val')
        test_data, test_loader = self._get_data(flag = 'test')

        loss_lambda = self.args.loss_lambda

        path = os.path.join(self.args.checkpoints, setting)
        if not os.path.exists(path):
            os.makedirs(path)

        time_now = time.time()
        
        train_steps = len(train_loader)
        early_stopping = EarlyStopping(patience=self.args.patience, verbose=True)
        
        model_optim = self._select_optimizer()
        criterion =  self._select_criterion()

        if self.args.use_amp:
            scaler = torch.cuda.amp.GradScaler()

        itera_train_loss_mse = []
        itera_train_loss_cl = []

        epoch_train_loss = []
        # epoch_train_loss_mse = []
        # epoch_train_loss_cl = []

        epo_eval_loss = []
        epo_test_loss = []

        for epoch in range(self.args.train_epochs):
            iter_count = 0
            train_loss = []
            
            self.model.train()
            epoch_time = time.time()
            for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(train_loader):
                iter_count += 1
                
                model_optim.zero_grad()
                pred, true, loss_mse, loss_cl = self._process_one_batch(
                    train_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
                
                loss = loss_mse * (1.0 - loss_lambda)
                itera_train_loss_mse.append(loss_mse.cpu().item())

                if loss_lambda > 0.0:
                    itera_train_loss_cl.append(loss_cl.cpu().item())
                    loss += loss_cl * loss_lambda


                train_loss.append(loss.item())
                
                if i % 100==0:
                    print("\titers: {0}, epoch: {1} | loss: {2:.7f}".format(i + 1, epoch + 1, loss.item()))
                    speed = (time.time()-time_now)/iter_count
                    left_time = speed*((self.args.train_epochs - epoch)*train_steps - i)
                    print('\tspeed: {:.4f}s/iter; left time: {:.4f}s'.format(speed, left_time))
                    iter_count = 0
                    time_now = time.time()
                
                if self.args.use_amp:
                    scaler.scale(loss).backward()
                    scaler.step(model_optim)
                    scaler.update()
                else:
                    loss.backward()
                    model_optim.step()
                    # print("backward")

            print("Epoch: {} cost time: {}".format(epoch+1, time.time()-epoch_time))
            train_loss = np.average(train_loss)
            vali_loss = self.vali(vali_data, vali_loader, criterion)
            test_loss = self.vali(test_data, test_loader, criterion)

            epoch_train_loss.append(train_loss)
            epo_eval_loss.append(vali_loss)
            epo_test_loss.append(test_loss)

            print("Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} Test Loss: {4:.7f}".format(
                epoch + 1, train_steps, train_loss, vali_loss, test_loss))
            early_stopping(vali_loss, self.model, path)
            if early_stopping.early_stop:
                print("Early stopping")
                break

            if self.args.cos_lr:
                adjust_learning_rate_cos(model_optim, epoch+1, self.args)
            else:
                adjust_learning_rate(model_optim, epoch+1, self.args)

            if self.args.closs_decay:
                loss_lambda = max(loss_lambda - 0.1, 0.0)
                print("Update contrastive loss weight to: ", str(loss_lambda))
            
        best_model_path = path+'/'+'checkpoint.pth'
        self.model.load_state_dict(torch.load(best_model_path))
        
        loss_record = {"itera_train_loss_mse": itera_train_loss_mse, "itera_train_loss_cl": itera_train_loss_cl,
                        "epoch_train_loss": epoch_train_loss, "epo_eval_loss": epo_eval_loss, "epo_test_loss": epo_test_loss, "num_step": train_steps}

        # result save
        folder_path = self.args.des_path + '/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        torch.save(loss_record, folder_path + "training_log.pt")

        return self.model

    def test(self, setting):
        test_data, test_loader = self._get_data(flag='test')
        
        self.model.eval()
        
        preds = []
        trues = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(test_loader):
            pred, true, _, _ = self._process_one_batch(
                test_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())
            trues.append(true.detach().cpu().numpy())

        preds = np.array(preds)
        trues = np.array(trues)
        print('test shape:', preds.shape, trues.shape)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        trues = trues.reshape(-1, trues.shape[-2], trues.shape[-1])
        print('test shape:', preds.shape, trues.shape)

        # result save
        folder_path = self.args.des_path + '/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)

        mae, mse, rmse, mape, mspe = metric(preds, trues)
        print('mse:{}, mae:{}'.format(mse, mae))

        np.save(folder_path+'metrics.npy', np.array([mae, mse, rmse, mape, mspe]))
        # np.save(folder_path+'pred.npy', preds)
        # np.save(folder_path+'true.npy', trues)

        return

    def predict(self, setting, load=False):
        pred_data, pred_loader = self._get_data(flag='pred')
        
        if load:
            path = os.path.join(self.args.checkpoints, setting)
            best_model_path = path+'/'+'checkpoint.pth'
            self.model.load_state_dict(torch.load(best_model_path))

        self.model.eval()
        
        preds = []
        
        for i, (batch_x,batch_y,batch_x_mark,batch_y_mark) in enumerate(pred_loader):
            pred, true, _ , _ = self._process_one_batch(
                pred_data, batch_x, batch_y, batch_x_mark, batch_y_mark)
            preds.append(pred.detach().cpu().numpy())

        preds = np.array(preds)
        preds = preds.reshape(-1, preds.shape[-2], preds.shape[-1])
        
        # result save
        folder_path = self.args.des_path + '/' + setting +'/'
        if not os.path.exists(folder_path):
            os.makedirs(folder_path)
        
        np.save(folder_path+'real_prediction.npy', preds)
        
        return

    def _process_one_batch(self, dataset_object, batch_x, batch_y, batch_x_mark, batch_y_mark):
        criterion =  self._select_criterion()
        
        batch_x = batch_x.float().to(self.device)
        batch_y = batch_y.float()

        batch_x_mark = batch_x_mark.float().to(self.device)
        batch_y_mark = batch_y_mark.float().to(self.device)

        # print("batch_x", batch_x.shape)
        # print("batch_x_mark", batch_x_mark.shape)

        contrast_loss = 0.0

        # decoder input
        if self.args.padding==0:
            dec_inp = torch.zeros([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        elif self.args.padding==1:
            dec_inp = torch.ones([batch_y.shape[0], self.args.pred_len, batch_y.shape[-1]]).float()
        dec_inp = torch.cat([batch_y[:,:self.args.label_len,:], dec_inp], dim=1).float().to(self.device)   # first label_len time stamps have actual values, pred_len time stamps have 0 values.
        # encoder - decoder
        if self.args.use_amp:
            with torch.cuda.amp.autocast():
                if self.args.model=='informer' or self.args.model=='informerstack':
                    if self.args.output_attention:
                        pred, out1, out2, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                    else:
                        pred, out1, out2 = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                elif 'moco' in self.args.model or "cost" in self.args.model:
                    pred, out1, contrast_loss  = self.model(batch_x, batch_x_mark)
                else:
                    pred, out1, out2 = self.model(batch_x)
        else:
            if self.args.model=='informer' or self.args.model=='informerstack':
                if self.args.output_attention:
                    pred, out1, out2, attn = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
                else:
                    pred, out1, out2 = self.model(batch_x, batch_x_mark, dec_inp, batch_y_mark)
            
            elif 'moco' in self.args.model or "cost" in self.args.model:
                    pred, _, contrast_loss  = self.model(batch_x, batch_x_mark)

            else:
                pred, out1, out2 = self.model(batch_x)
                
        if self.args.inverse:
            pred = dataset_object.inverse_transform(pred)

        f_dim = -1 if self.args.features=='MS' else 0
        batch_y = batch_y[:,-self.args.pred_len:,f_dim:].to(self.device)

        loss_mse = criterion(pred, batch_y)

        if self.args.loss_lambda > 0.0 and 'moco' not in self.args.model and "cost" not in self.args.model:
            contrast_loss = hierarchical_contrastive_loss(out1, out2, self.args.l2norm)

        return pred, batch_y, loss_mse, contrast_loss
