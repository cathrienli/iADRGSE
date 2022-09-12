# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
from sklearn.model_selection import train_test_split
from model import Transformer

# multi-GPU option
# os.environ['CUDA_VISIBLE_DEVICES'] = '1'

from time import time
import pandas
import pandas as pd
import torch.optim as optim
from collections import OrderedDict
import argparse
import logging
import os
import random
from sklearn.model_selection import KFold, StratifiedKFold
import torch.utils.data as Data
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, \
    average_precision_score
import copy
import pywt
import math
from tensorflow import keras


EMB_INIT_EPS = 2.0
gamma = 12.0


# --------------------------------------------initial param------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run iADR.")

    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed.')
    parser.add_argument('--data_dir', nargs='?', default='./data/',
                        help='Input data path.')
    parser.add_argument('--save_dir', nargs='?', default='./model/',
                        help='Input data path.')
    parser.add_argument('--log_dir', nargs='?', default='./log/',
                        help='Input data path.')
    parser.add_argument('--fp_file', nargs='?', default='./data/fp(256).txt',
                        help='Input data path.')
    parser.add_argument('--label_file', nargs='?', default='./data/labels(2248).csv',
                        help='Input data path.')
    parser.add_argument('--infomax_file', nargs='?', default='./data/pretain_smiles_GIN_infomax.npy',
                        help='Input data path.')
    parser.add_argument('--edgepred_file', nargs='?', default='./data/pretain_smiles_GIN_edgepred.npy',
                        help='Input data path.')
    parser.add_argument('--contextpred_file', nargs='?', default='./data/pretain_smiles_GIN_contextpred.npy',
                        help='Input data path.')
                        
                        
    #test
    # parser.add_argument('--test_fp_file', nargs='?', default='./data/fp(833).txt',
    #                     help='Input data path.')
    # parser.add_argument('--test_label_file', nargs='?', default='./data/label(27).csv',
    #                     help='Input data path.')
    # parser.add_argument('--test_infomax_file', nargs='?', default='./data/gin_infomax_drug(832).npy',
    #                     help='Input data path.')
    # parser.add_argument('--test_edgepred_file', nargs='?', default='./data/gin_edgepred_drug(832).npy',
    #                     help='Input data path.')
    # parser.add_argument('--test_contextpred_file', nargs='?', default='./data/gin_contextpred_drug(832).npy',
    #                     help='Input data path.')


    parser.add_argument('--DDI_batch_size', type=int, default=1024,
                        help='DDI batch size.')
    parser.add_argument('--conv_dim_list', nargs='?', default='[64, 32, 16]',
                        help='Output sizes of every aggregation layer.')

    parser.add_argument('--dropout', type=float, default=0.5,
                        help='Dropout probability w.r.t. message dropout for each deep layer. 0: no dropout.')
    parser.add_argument('--weight_decay', type=float, default=1e-3,
                        help='Lambda when calculating l2 loss.')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate.')
    parser.add_argument('--alpha', type=float, default=1e-2,
                        help='Alpha for the leaky_relu.')
    parser.add_argument('--n_epoch', type=int, default=2000,
                        help='Number of epoch.')
    parser.add_argument('--stopping_steps', type=int, default=30,
                        help='Number of epoch for early stopping')
    parser.add_argument('--evaluate_every', type=int, default=1,
                        help='Epoch interval of evaluating DDI.')
    parser.add_argument('--multi_type', type=bool, default=False,
                        help='whether task is multi-class')
    parser.add_argument('--n_hidden_1', type=int, default=2048,
                        help='FC hidden 1 dim')
    parser.add_argument('--n_hidden_2', type=int, default=2048,
                        help='FC hidden 2 dim')
    parser.add_argument('--out_dim', type=int, default=27,
                        help='FC output dim')
    parser.add_argument('--cuda', type=bool, default=True,
                        help='GPU or CPU')

    args = parser.parse_args()

    return args


# ----------------------------------------define log information--------------------------------------------------------

# create log information
def create_log_id(dir_path):
    log_count = 0
    file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    while os.path.exists(file_path):
        log_count += 1
        file_path = os.path.join(dir_path, 'log{:d}.log'.format(log_count))
    return log_count


def logging_config(folder=None, name=None,
                   level=logging.DEBUG,
                   console_level=logging.DEBUG,
                   no_console=True):
    if not os.path.exists(folder):
        os.makedirs(folder)
    for handler in logging.root.handlers:
        logging.root.removeHandler(handler)
    logging.root.handlers = []
    logpath = os.path.join(folder, name + ".txt")
    print("All logs will be saved to %s" % logpath)

    logging.root.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    logfile = logging.FileHandler(logpath)
    logfile.setLevel(level)
    logfile.setFormatter(formatter)
    logging.root.addHandler(logfile)

    if not no_console:
        logconsole = logging.StreamHandler()
        logconsole.setLevel(console_level)
        logconsole.setFormatter(formatter)
        logging.root.addHandler(logconsole)
    return folder


# -----------------------------------------loading KG data and DDI 5-fold data------------------------------------------

# loading data


def chunkIt(seq, num):
    data = []
    for i in range(0, len(seq), num):
        if i + num > len(seq):
            data.append(seq[i:])
        else:
            data.append(seq[i:i + num])

    return data


def early_stopping(model, epoch, best_epoch, train_auc, best_auc, bad_counter):
    if train_auc > best_auc:
        best_auc = train_auc
        bad_counter = 0
        save_model(model, args.save_dir, epoch, best_epoch)
        best_epoch = epoch
    else:
        bad_counter += 1
    return bad_counter, best_auc, best_epoch


def save_model(model, model_dir, current_epoch, last_best_epoch=None):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(current_epoch))
    torch.save({'model_state_dict': model.state_dict(), 'epoch': current_epoch}, model_state_file)


    if last_best_epoch is not None and current_epoch != last_best_epoch:
        old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
        if os.path.exists(old_model_state_file):
            os.system('rd {}'.format(old_model_state_file))


def load_model(model, model_dir, best_epoch):
    model_path = os.path.join(model_dir, 'model_epoch{}.pth'.format(best_epoch))
    checkpoint = torch.load(model_path, map_location=get_device(args))

    try:
        model.load_state_dict(checkpoint['model_state_dict'])
    except RuntimeError:
        state_dict = OrderedDict()
        for k, v in checkpoint['model_state_dict'].items():
            k_ = k[7:]  # remove 'module.' of DistributedDataParallel instance
            state_dict[k_] = v
        model.load_state_dict(state_dict)

    model.eval()
    return model


def _L2_loss_mean(x):
    return torch.mean(torch.sum(torch.pow(x, 2), dim=1, keepdim=False) / 2.)


def get_device(args):
    args.gpu = False
    if torch.cuda.is_available() and args.cuda:
        args.gpu = True
    device = torch.device("cuda:0" if args.gpu else "cpu")
    return device


# ----------------------------------------------  Main model part  -----------------------------------------------------
class ADRModel(nn.Module):

    def __init__(self, device):
        super(ADRModel, self).__init__()
        print("transformer initial...")
        dim = 8
        self.liner = nn.Linear(27 * 4, 27)
        self.gin_liner = nn.Sequential(nn.BatchNorm1d(27*4),
                                          nn.Dropout(args.dropout),
                                          nn.Linear(27*4, 27))
        self.wavelet_liner = nn.Sequential(nn.Linear(128, 27))
        self.emb = nn.Embedding(20, dim)
        self.atten = nn.Parameter(torch.randn(27, dtype=torch.float32), requires_grad=True)
        self.softmax = nn.Softmax(dim=1)
        self.act = nn.LeakyReLU()
        self.norm = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(args.dropout)
        self.trans = Transformer(
            d_model=dim,
            dropout=args.dropout,
            nhead=2,
            dim_feedforward=128,
            num_encoder_layers=1,
            num_decoder_layers=1
        )
        self.layer = nn.Sequential(nn.BatchNorm1d(2048),
                                  nn.Linear(2048, 256),
                                  nn.LeakyReLU(),
                                  nn.Dropout(args.dropout),
                                  nn.BatchNorm1d(256),
                                  nn.Linear(256, 27))
        self.device = device

    def init_weights(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs, fpts, wavelets):  # inputs (seq_len, batch, dim: 300, size, 4), fpts (size, 256)
        tgt = self.emb(fpts).permute(1, 0, 2) # 256, size, 32
        tgt = self.dropout(self.norm(tgt))
        output = self.trans(None, tgt) # size, 256, 32
        outputs3 = output.permute(1, 0, 2).flatten(1)
        outputs1 = self.layer(outputs3)
        outputs2 = torch.cat((inputs, outputs1), 1)
        outputs = self.liner(outputs2)
        return outputs




# -------------------------------------- metrics and evaluation define -------------------------------------------------
def Accuracy_sample(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        p = sum(np.logical_not(np.logical_xor(y_true[i], y_pred[i])))
        count += p / y_true.shape[1]
    return count / y_true.shape[0]


def Accuracy_micro(y_true, y_pred):
    count = 0
    for i in range(y_true.shape[0]):
        count += sum(np.logical_not(np.logical_xor(y_true[i], y_pred[i])))
    return count / y_true.size


def Accuracy_macro(y_true, y_pred):
    print(y_true.shape, y_pred.shape)
    count = 0
    for j in range(y_true.shape[1]):
        count += (1 - sum(np.logical_xor(y_true[:, j], y_pred[:, j])) / y_true.shape[0])
    return count / y_true.shape[1]

def calc_metrics(y_true, pred_score):
    y_true = y_true.cpu().detach().numpy()
    # y_pred = pred_score.cpu().detach().numpy()
    y_pred = copy.deepcopy(pred_score)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    acc = Accuracy_macro(y_true, y_pred)
    precision = precision_score(y_true, y_pred, average='macro')
    recall = recall_score(y_true, y_pred, average='macro')
    f1 = f1_score(y_true, y_pred, average='macro')
    aupr = average_precision_score(y_true, pred_score, average='macro')
    auc = roc_auc_score(y_true, pred_score,average='macro')
    return acc, precision, recall, f1,auc, aupr
    
    
def calc_metrics2(y_true, pred_score):
    m1 = keras.metrics.BinaryAccuracy(name='accuracy')
    m2 = keras.metrics.Precision(name='precision')
    m3 = keras.metrics.Recall(name='recall')
    m4 = keras.metrics.AUC(name='AUC',multi_label=True,num_labels=27,num_thresholds=498)
    m5 = keras.metrics.AUC(name='AUPR',curve='PR',multi_label=True,num_labels=27,num_thresholds=498)
    y_true = y_true.cpu().detach().numpy()
    acc = m1.update_state(y_true,pred_score)
    precision = m2.update_state(y_true,pred_score)
    recall = m3.update_state(y_true,pred_score)
    aupr = m4.update_state(y_true,pred_score)
    auc = m5.update_state(y_true,pred_score)
    acc = m1.result().numpy()
    precision = m2.result().numpy()
    recall = m3.result().numpy()
    auc = m4.result().numpy()
    aupr = m5.result().numpy()
    return acc, precision, recall, auc, aupr


def evaluate2(args, model, test_x, test_fpt, test_wavelet, test_y):
    model.eval()

    with torch.no_grad():
        outputs = model(test_x, test_fpt, test_wavelet)
        # print(outputs.shape)
        out = torch.sigmoid(outputs).cpu().detach().numpy()
        # print(out.shape)
        y_pred = copy.deepcopy(out)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        loss = loss_func(outputs, test_y.float())
        loss_list = np.sum(loss.cpu().detach().numpy(), axis=1)
        # print(loss_list.shape)
        acc, precision, recall, auc, aupr = calc_metrics2(test_y, out)
        return acc, precision, recall, auc, aupr
        
def evaluate(args, model, test_x, test_fpt, test_wavelet, test_y):
    model.eval()

    with torch.no_grad():
        outputs = model(test_x, test_fpt, test_wavelet)
        # print(outputs.shape)
        out = torch.sigmoid(outputs).cpu().detach().numpy()
        # print(out.shape)
        y_pred = copy.deepcopy(out)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        loss = loss_func(outputs, test_y.float())
        loss_list = np.sum(loss.cpu().detach().numpy(), axis=1)
        # print(loss_list.shape)
        acc, precision, recall, f1,auc, aupr = calc_metrics(test_y, out)
        return acc, precision, recall, f1,auc, aupr
            


def wavelet_encoder(seq):
    meta_drug = np.array(list(map(lambda x: int(x, 16), seq)))
    ca, cd = pywt.dwt(meta_drug, 'db1')
    drug_feature = np.divide(ca, (np.sum(ca)+1e-12)).astype(np.float32)
    return drug_feature


def trans_fpt(fpts):
    fpt_feature = np.zeros((fpts.shape[0], 128))
    index = 0
    for fpt in fpts:
        hex_arr = wavelet_encoder(fpt)
        fpt_feature[index] = hex_arr
        index += 1
    return fpt_feature





# -----------------------------------   train model  -------------------------------------------------------------------

def train(args):
    # seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    # set log file
    log_save_id = create_log_id(args.log_dir)
    logging_config(folder=args.log_dir, name='log{:d}'.format(log_save_id), no_console=False)
    logging.info(args)

    # GPU / CPU
    device = get_device(args)
    # initialize data
    data_fp = pd.read_csv(args.fp_file).values.astype(np.int32)
    wavelet_fp = trans_fpt(data_fp.astype(np.str_))
    y = pd.read_csv(args.label_file)
    infomax = np.load(args.infomax_file)
    edge = np.load(args.edgepred_file)
    con = np.load(args.contextpred_file)
    features = np.concatenate((infomax, edge, con), axis=1).astype(np.float32)
    
    x_train,x_test,y_train,y_test= train_test_split(features,y,test_size=0.1,random_state=2021)
    data_fp_train,data_fp_test,y_train,y_test = train_test_split(data_fp,y,test_size=0.1,random_state=2021)

    #test
    # test_data_fp = pd.read_csv(args.test_fp_file,sep=',',header=0,index_col=0).values.astype(np.int32)
    # test_wavelet_fp = trans_fpt(test_data_fp.astype(np.str_))
    # y_test = pd.read_csv(args.test_label_file,sep=',',index_col=0).values.astype(np.int32)
    # # y_test = y_test[:,:24]
    # test_infomax = np.load(args.test_infomax_file)
    # test_edge = np.load(args.test_edgepred_file)
    # test_con = np.load(args.test_contextpred_file)
    # test_features = np.concatenate((test_infomax, test_edge, test_con), axis=1).astype(np.float32)




    # features = features.transpose(2, 0, 1)  # 27, size, 4

    train_graph = None

    all_acc_list = []
    all_precision_list = []
    all_recall_list = []
    all_aupr_list = []
    all_auc_list = []
    all_loss_list = np.array([[10000, 0, 0, 10000]])

    all_macro_precision_list = []
    all_macro_recall_list = []
    all_macro_auc_list = []
    all_macro_aupr_list = []
    all_micro_precision_list = []
    all_micro_recall_list = []
    all_micro_auc_list = []
    all_micro_aupr_list = []
    
    all_pred_list = []
    start_t = time()
    # train model
    
    train_y = torch.from_numpy(y_train.values)
    train_x = torch.from_numpy(x_train)
    test_x = torch.from_numpy(x_test)
    test_y = torch.from_numpy(y_test.values)

    train_fpt = torch.from_numpy(data_fp_train)
    test_fpt = torch.from_numpy(data_fp_test)
    train_wavelet = torch.FloatTensor(wavelet_fp)
    test_wavelet = torch.FloatTensor(wavelet_fp)


    # train_y = torch.from_numpy(y.values)
    # train_x = torch.from_numpy(features)
    # test_x = torch.from_numpy(test_features)
    # test_y = torch.from_numpy(y_test)

    # train_fpt = torch.from_numpy(data_fp)
    # test_fpt = torch.from_numpy(test_data_fp)
    # train_wavelet = torch.FloatTensor(wavelet_fp)
    # test_wavelet = torch.FloatTensor(wavelet_fp)


    if args.gpu:
            train_x = train_x.to(device)
            train_y = train_y.to(device)
            test_x = test_x.to(device)
            test_y = test_y.to(device)
            train_fpt = train_fpt.to(device)
            test_fpt = test_fpt.to(device)
            train_wavelet = train_wavelet.to(device)
            test_wavelet = test_wavelet.to(device)

    torch_dataset = data.TensorDataset(train_x, train_fpt, train_y)
    loader = data.DataLoader(dataset=torch_dataset,batch_size=128, shuffle=True, num_workers=1  )

    model = ADRModel(device)
    model.init_weights()
    model.to(device)
        # logging.info(model)

        # define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        # loss_func = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')
    loss_func = torch.nn.BCEWithLogitsLoss()

    macro_precision_list = []
    macro_recall_list = []
    macro_auc_list = []
    macro_aupr_list = []

    bad_counter = 0
    best_auc = 0
    best_epoch = 0

    time0 = time()
    for epoch in range(1, args.n_epoch + 1):
        out_list = np.empty((0, 27))
        loss_sum = 0
        batch_size = args.DDI_batch_size
        batch = math.ceil(train_y.shape[0] / batch_size)
        for batch_id in range(batch):
            batch_x, batch_fpt, batch_y = torch_dataset[batch_id*batch_size : (batch_id+1)*batch_size]
            model.train()
            out = model(batch_x, batch_fpt, train_wavelet)
            loss = loss_func(out, batch_y.float())
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            out_list = np.concatenate((out_list, torch.sigmoid(out).cpu().detach().numpy()), axis=0)
            loss_sum += loss.item()

        train_acc2, train_precision2, train_recall2, train_auc2, train_aupr2 = calc_metrics2(train_y, out_list)
            # train_auc = roc_auc_score(train_y.cpu().detach().numpy(), out_list, average='macro')
            # logging.info('DDI Training: Folder {:04d} | Epoch {:04d} | AUC {:.4f} | Loss {:.4f}'.format(folder, epoch, train_auc, loss_sum))
        logging.info(
            'Training:Epoch {:04d} | ACC {:.4f} Precision {:.4f} Recall {:.4f} AUC {:.4f}'
            'AUPR {:.4f}| Loss {:.4f}'.format(epoch, train_acc2,train_precision2, train_recall2,train_auc2, train_aupr2,loss.item()))
                                                         
        train_acc, train_precision, train_recall, train_f1,train_auc, train_aupr = calc_metrics(train_y, out_list)
            # train_auc = roc_auc_score(train_y.cpu().detach().numpy(), out_list, average='macro')
            # logging.info('DDI Training: Folder {:04d} | Epoch {:04d} | AUC {:.4f} | Loss {:.4f}'.format(folder, epoch, train_auc, loss_sum))
        logging.info('Training:Epoch {:04d} | ACC {:.4f} Precision {:.4f} Recall {:.4f} F1_score {:.4f} AUC {:.4f}'
            'AUPR {:.4f}'.format(epoch, train_acc,train_precision, train_recall,train_f1,train_auc, train_aupr))

        bad_counter, best_auc, best_epoch = early_stopping(model, epoch, best_epoch, train_auc, best_auc, bad_counter)
        if bad_counter >= args.stopping_steps or epoch == args.n_epoch:
            model = load_model(model, args.save_dir, best_epoch)
            # out_pred = model(test_x, test_fpt, test_wavelet)
            # pred_score = torch.sigmoid(out_pred.squeeze(0)).cpu().detach().numpy()
            # all_pred_list.append(pred_score)
            acc2,precision2, recall2, auc2,aupr2 = evaluate2(args, model, test_x, test_fpt, test_wavelet, test_y)
            logging.info(
                'Final Evaluation:| Total Time {:.1f}s | ACC {:.4f} Precision {:.4f} Recall {:.4f} AUC {:.4f} '
                'AUPR {:.4f} '.format(time() - time0, acc2,precision2, recall2, auc2, aupr2))
                
            acc,precision, recall,f1, auc,aupr = evaluate(args, model, test_x, test_fpt, test_wavelet, test_y)
            logging.info(
                'Final Evaluation:| Total Time {:.1f}s | ACC {:.4f} Precision {:.4f} Recall {:.4f} F1_score {:.4f} AUC {:.4f} '
                'AUPR {:.4f} '.format(time() - time0, acc,precision, recall,f1, auc, aupr))
            break




if __name__ == '__main__':
    args = parse_args()
    train(args)
