import argparse
import os
import torch

from exp.exp_informer import Exp_Informer
torch.autograd.set_detect_anomaly(True)

parser = argparse.ArgumentParser(description='[Informer] Long Sequences Forecasting')

parser.add_argument('--model', type=str, required=True, default='informer',help='model of experiment, options: [informer, lstm, tcn]')

parser.add_argument('--data', type=str, required=True, default='ETTh1', help='data')
parser.add_argument('--root_path', type=str, default='./data/ETT/', help='root path of the data file')
parser.add_argument('--data_path', type=str, default='ETTh1.csv', help='data file')    
parser.add_argument('--features', type=str, default='M', help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')
parser.add_argument('--target', type=str, default='OT', help='target feature in S or MS task')
parser.add_argument('--freq', type=str, default='h', help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
parser.add_argument('--checkpoints', type=str, default='./checkpoints/', help='location of model checkpoints')

parser.add_argument('--seq_len', type=int, default=96, help='input sequence length of Informer encoder')
parser.add_argument('--label_len', type=int, default=48, help='start token length of Informer decoder')
parser.add_argument('--pred_len', type=int, default=24, help='prediction sequence length')
# Informer decoder input: concat[start token series(label_len), zero padding series(pred_len)]

parser.add_argument('--enc_in', type=int, default=7, help='encoder input size')
parser.add_argument('--dec_in', type=int, default=7, help='decoder input size')
parser.add_argument('--c_out', type=int, default=7, help='output size')
parser.add_argument('--d_model', type=int, default=320, help='dimension of model')
parser.add_argument('--n_heads', type=int, default=8, help='num of heads')
parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
parser.add_argument('--s_layers', type=str, default='3,2,1', help='num of stack encoder layers')
parser.add_argument('--d_ff', type=int, default=2048, help='dimension of fcn')
parser.add_argument('--factor', type=int, default=5, help='probsparse attn factor')
parser.add_argument('--padding', type=int, default=0, help='padding type')
parser.add_argument('--distil', action='store_false', help='whether to use distilling in encoder, using this argument means not using distilling', default=True)
parser.add_argument('--dropout', type=float, default=0.05, help='dropout')
parser.add_argument('--attn', type=str, default='prob', help='attention used in encoder, options:[prob, full]')
parser.add_argument('--embed', type=str, default='learned', help='time features encoding, options:[timeF, fixed, learned]')
parser.add_argument('--activation', type=str, default='gelu',help='activation')
parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
parser.add_argument('--mix', action='store_false', help='use mix attention in generative decoder', default=True)
parser.add_argument('--cols', type=str, nargs='+', help='certain cols from the data files as the input features')
parser.add_argument('--num_workers', type=int, default=0, help='data loader num workers')
parser.add_argument('--itr', type=int, default=2, help='experiments times')
parser.add_argument('--train_epochs', type=int, default=6, help='train epochs')
parser.add_argument('--batch_size', type=int, default=32, help='batch size of train input data')
parser.add_argument('--patience', type=int, default=3, help='early stopping patience')
parser.add_argument('--learning_rate', type=float, default=0.0001, help='optimizer learning rate')
parser.add_argument('--des', type=str, default='test',help='exp description')
parser.add_argument('--loss', type=str, default='mse',help='loss function')
parser.add_argument('--lradj', type=str, default='type1',help='adjust learning rate')
parser.add_argument('--use_amp', action='store_true', help='use automatic mixed precision training', default=False)
parser.add_argument('--inverse', action='store_true', help='inverse output data', default=False)

parser.add_argument('--des_path', type=str, default="training",  help='Save path')
parser.add_argument('--use_gpu', type=bool, default=True, help='use gpu')
parser.add_argument('--gpu', type=int, default=0, help='gpu')
parser.add_argument('--use_multi_gpu', action='store_true', help='use multiple gpus', default=False)
parser.add_argument('--devices', type=str, default='0,1,2,3',help='device ids of multile gpus')

#contrastive
parser.add_argument('--loss_lambda', type=float, default=0.5, help='coefficient between MSE loss and contrastive loss')
parser.add_argument('--mask_rate', type=float, default=0.5, help='optimizer learning rate')
parser.add_argument('--l2norm', type=str, default="False", help='whether to use l2norm in contrastive loss')

# TSN
parser.add_argument('--kernel_size', type=int, default=3, help='kernel_size in TCN')

# MoCo
parser.add_argument('--moco_average_pool', type=str, default="True", help='whether to use average pool in TCN-MoCo contrastive loss')
parser.add_argument('--data_aug', type=str, default=None, help='data augmentation method')
parser.add_argument('--cos_lr', type=str, default="False", help='whether to use cosine learning rate schedular')
parser.add_argument('--mare', type=str, default="False", help='whether to use Mixture of Auto-regressive Experts')
parser.add_argument('--time_feature_embed', type=str, default="False", help='whether to use time feature embedding layer')
parser.add_argument('--closs_decay', type=str, default="False", help='whether to decay weight of cl loss')

args = parser.parse_args()

args.use_gpu = True if torch.cuda.is_available() and args.use_gpu else False

def bool_change(x):
    if x == "False":
        x= False
    else:
        x = True
    return x

args.l2norm = bool_change(args.l2norm)
args.moco_average_pool = bool_change(args.moco_average_pool)
args.cos_lr = bool_change(args.cos_lr)
args.mare = bool_change(args.mare)
args.time_feature_embed = bool_change(args.time_feature_embed)
args.closs_decay = bool_change(args.closs_decay)

print("l2norm", args.l2norm)
print("moco_average_pool", args.moco_average_pool)

if args.use_gpu == False:
    print("exit")
    exit() 

if args.use_gpu and args.use_multi_gpu:
    args.devices = args.devices.replace(' ','')
    device_ids = args.devices.split(',')
    args.device_ids = [int(id_) for id_ in device_ids]
    args.gpu = args.device_ids[0]

data_parser = {
    'ETTh1':{'data':'ETTh1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTh2':{'data':'ETTh2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm1':{'data':'ETTm1.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'ETTm2':{'data':'ETTm2.csv','T':'OT','M':[7,7,7],'S':[1,1,1],'MS':[7,7,1]},
    'WTH':{'data':'WTH.csv','T':'WetBulbCelsius','M':[12,12,12],'S':[1,1,1],'MS':[12,12,1]},
    'ECL':{'data':'ECL.csv','T':'MT_320','M':[321,321,321],'S':[1,1,1],'MS':[321,321,1]},
    'Solar':{'data':'solar_AL.csv','T':'POWER_136','M':[137,137,137],'S':[1,1,1],'MS':[137,137,1]},
}
if args.data in data_parser.keys():
    data_info = data_parser[args.data]
    args.data_path = data_info['data']
    args.target = data_info['T']
    args.enc_in, args.dec_in, args.c_out = data_info[args.features]

args.s_layers = [int(s_l) for s_l in args.s_layers.replace(' ','').split(',')]
args.detail_freq = args.freq
args.freq = args.freq[-1:]

print('Args in experiment:')
print(args)

orignial_args = args

Exp = Exp_Informer

for ii in range(args.itr):
    args = orignial_args
    print('Args in experiment:')
    print(args)
    # setting record of experiments
    if args.model=='informer' or args.model=='informerstack':
        setting = 'contrast-{}_{}_contrastL{}_mask{}_l2norm{}_ft{}_sl{}_ll{}_pl{}_dm{}_nh{}_el{}_dl{}_df{}_at{}_fc{}_eb{}_dt{}_mx{}_{}_{}'.format(
                args.model, args.data, args.loss_lambda, 
                args.mask_rate, str(args.l2norm), args.features, 
                args.seq_len, args.label_len, args.pred_len,
                args.d_model, args.n_heads, args.e_layers, args.d_layers, args.d_ff, args.attn, args.factor, 
                args.embed, args.distil, args.mix, args.des, ii)
    elif args.model=='lstm' or args.model=='tcn' or args.model=='dtcn' or args.model == 'informer-encoder':
        setting = 'contrast-{}_{}_contrastL{}_mask{}_l2norm{}_ft{}_sl{}_pl{}_dm{}_el{}_{}_{}'.format(
                args.model, args.data, args.loss_lambda, 
                args.mask_rate, str(args.l2norm), args.features, 
                args.seq_len, args.pred_len,
                args.d_model, args.e_layers, args.des, ii)
    elif "moco" in args.model or args.model=="cost-e2e":
        setting = 'contrast-{}_{}_contrastL{}_mask{}_l2norm{}_ft{}_timeF{}_mare{}_cldecay{}_sl{}_pl{}_dm{}_el{}_avg{}_cos{}_aug{}_{}_{}'.format(
                args.model, args.data, args.loss_lambda, 
                args.mask_rate, str(args.l2norm), args.features, str(args.time_feature_embed), str(args.mare), str(args.closs_decay),
                args.seq_len, args.pred_len,
                args.d_model, args.e_layers, str(args.moco_average_pool), str(args.cos_lr), args.data_aug, args.des, ii)
    else:
        print("The model name is not right.")
        exit()


    exp = Exp(args) # set experiments
    print('>>>>>>>start training : {}>>>>>>>>>>>>>>>>>>>>>>>>>>'.format(setting))
    exp.train(setting)
    
    print('>>>>>>>testing : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
    exp.test(setting)

    if args.do_predict:
        print('>>>>>>>predicting : {}<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<'.format(setting))
        exp.predict(setting, True)

    torch.cuda.empty_cache()
