# coding=utf-8
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data

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


EMB_INIT_EPS = 2.0
gamma = 12.0


# --------------------------------------------initial param------------------------------------------------------------

def parse_args():
    parser = argparse.ArgumentParser(description="Run iADR.")

    parser.add_argument('--seed', type=int, default=2022,
                        help='Random seed.')
    parser.add_argument('--data_dir', nargs='?', default='../data/',
                        help='Input data path.')
    parser.add_argument('--save_dir', nargs='?', default='../model/',
                        help='Input data path.')
    parser.add_argument('--log_dir', nargs='?', default='../log/',
                        help='Input data path.')
    parser.add_argument('--fp_file', nargs='?', default='../data/fp(256).txt',
                        help='Input data path.')
    parser.add_argument('--label_file', nargs='?', default='../data/labels(2248).csv',
                        help='Input data path.')
    parser.add_argument('--attentivefp_file', nargs='?', default='../data/drugs_structure_attentivefp_ToxCast.npy',
                        help='Input data path.')
    
    parser.add_argument('--infomax_file', nargs='?', default='../data/pretain_smiles_GIN_infomax.npy',
                        help='Input data path.')
    parser.add_argument('--edgepred_file', nargs='?', default='../data/pretain_smiles_GIN_edgepred.npy',
                        help='Input data path.')
    parser.add_argument('--contextpred_file', nargs='?', default='../data/pretain_smiles_GIN_contextpred.npy',
                        help='Input data path.')

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
    parser.add_argument('--stopping_steps', type=int, default=300,
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

    # file_name = os.path.join(model_dir, 'drug_embed{}.npy'.format(current_epoch))
    # np.save(file_name, all_embed.cpu().detach().numpy())
    #
    # data = np.load(file_name)
    # print(data.shape)
    #
    # if last_best_epoch is not None and current_epoch != last_best_epoch:
    #     old_model_state_file = os.path.join(model_dir, 'model_epoch{}.pth'.format(last_best_epoch))
    #     old_embedding_file = os.path.join(model_dir, 'drug_embed{}.npy'.format(last_best_epoch))
    #     if os.path.exists(old_model_state_file):
    #         os.system('rm {}'.format(old_model_state_file))
    #     if os.path.exists(old_embedding_file):
    #         os.system('rm {}'.format(old_embedding_file))

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
        # self.attention = nn.Parameter(torch.Tensor(300, 1))
        # self.attention.data.normal_(0, 0.1)
        # self.rnn_layer = nn.LSTM(4, 2, 2, bidirectional=True)
        print("transformer initial...")
        dim = 8
        self.liner = nn.Linear(27 * 4, 27)
        self.gin_liner = nn.Sequential(nn.BatchNorm1d(27*4),
                                           nn.Dropout(args.dropout),
                                           nn.Linear(27*4, 27))
        self.wavelet_liner = nn.Sequential(# nn.BatchNorm1d(128),
                                           # nn.Dropout(args.dropout),
                                           nn.Linear(128, 27))
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
        # for m in self.modules():
        #     if isinstance(m, nn.Linear):
        #         nn.init.normal_(m.weight.data, std=0.1)
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, inputs, fpts, wavelets):  # inputs (seq_len, batch, dim: 300, size, 4), fpts (size, 256)
        # att = torch.matmul(inputs, self.attention).permute(0, 2, 1)
        # agg = torch.matmul(att, inputs)

        # inputs, hidden = self.rnn_layer(inputs)
        # inputs = inputs.permute(1, 2, 0)
        # inputs = self.liner(inputs) # 300, size, 32
        # size = inputs.shape[0]
        # atten_w = self.softmax(inputs.matmul(self.atten))
        # inputs = atten_w.unsqueeze(1).bmm(inputs).squeeze(1)
        # inputs = self.gin_liner(inputs)
        # wavelets = self.wavelet_liner(wavelets)
        tgt = self.emb(fpts).permute(1, 0, 2) # 256, size, 8
        tgt = self.dropout(self.norm(tgt))
        output = self.trans(None, tgt) # size, 256, 8
        outputs3 = output.permute(1, 0, 2).flatten(1)#256*8=2048
        outputs1 = self.layer(outputs3)#feed forward 2048 256 27
        outputs2 = torch.cat((inputs, outputs1), 1)#27*3+27=108
        outputs = self.liner(outputs2)#108 27
        return outputs,outputs1




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
    count = 0
    for j in range(y_true.shape[1]):
        count += (1 - sum(np.logical_xor(y_true[:, j], y_pred[:, j])) / y_true.shape[0])
    return count / y_true.shape[1]


def calc_metrics(y_true, pred_score, multi_type):
    y_true = y_true.cpu().detach().numpy()
    y_pred = copy.deepcopy(pred_score)
    y_pred[y_pred >= 0.5] = 1
    y_pred[y_pred < 0.5] = 0
    if multi_type:
        acc = accuracy_score(y_true, y_pred)
        macro_precision = precision_score(y_true, y_pred, average='macro')
        macro_recall = recall_score(y_true, y_pred, average='macro')
        macro_auc = roc_auc_score(y_true, y_pred, average='macro')
        macro_aupr = average_precision_score(y_true, y_pred, average='macro')
        micro_precision = precision_score(y_true, y_pred, average='micro')
        micro_recall = recall_score(y_true, y_pred, average='micro')
        micro_auc = roc_auc_score(y_true, y_pred, average='micro')
        micro_aupr = average_precision_score(y_true, y_pred, average='micro')
        return acc, macro_precision, macro_recall, macro_auc, macro_aupr, micro_precision, micro_recall, micro_auc, micro_aupr
    else:
        acc = Accuracy_macro(y_true, y_pred)
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        aupr = average_precision_score(y_true, pred_score, average='macro')
        auc = roc_auc_score(y_true, pred_score, average='macro')
        return acc, precision, recall, aupr, auc, y_pred


def evaluate(args, model, test_x, test_fpt, test_wavelet, test_y, test_index):
    model.eval()

    with torch.no_grad():
        outputs = model(test_x, test_fpt, test_wavelet)
        out = torch.sigmoid(outputs).cpu().detach().numpy()
        y_pred = copy.deepcopy(out)
        y_pred[y_pred >= 0.5] = 1
        y_pred[y_pred < 0.5] = 0
        loss_func = torch.nn.BCEWithLogitsLoss(reduction='none')
        loss = loss_func(outputs, test_y.float())
        loss_list = np.sum(loss.cpu().detach().numpy(), axis=1)
        if not args.multi_type:
            acc, precision, recall, aupr, auc, y_pred = calc_metrics(test_y, out, args.multi_type)
            loss_list = np.concatenate((loss_list[:, np.newaxis], y_pred, test_y.cpu().detach().numpy(), test_index[:, np.newaxis]), axis=1)
            return precision, recall, aupr, acc, auc, loss_list
        else:
            prediction = torch.max(out, 1)[1]
            prediction = prediction.cuda().data.cpu().numpy()
            acc, macro_precision, macro_recall, macro_auc, macro_aupr, micro_precision, micro_recall, micro_auc, micro_aupr = calc_metrics(
                test_y, prediction, out, args.multi_type)
            # print(acc, macro_precision, macro_recall, macro_f1, micro_precision, micro_recall, micro_f1)
            return macro_precision, macro_recall, macro_auc, macro_aupr, micro_precision, micro_recall, micro_auc, micro_aupr, acc


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
    data_att = np.load(args.attentivefp_file)
    masking = np.load(args.masking_file)
    infomax = np.load(args.infomax_file)
    edge = np.load(args.edgepred_file)
    con = np.load(args.contextpred_file)
    # attentivefp = np.load('../data/pretain_smiles_MPNN_attentivefp.npy')
    # canonical = np.load('../data/pretain_smiles_MPNN_canonical.npy')

    features = np.concatenate((infomax, edge, con), axis=1).astype(np.float32)
    # features = features.transpose(2, 0, 1)  # 27, size, 4

    train_graph = None

    all_acc_list = []
    all_precision_list = []
    all_recall_list = []
    all_aupr_list = []
    all_auc_list = []
    all_loss_list = np.array([[10000, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 10000]])

    all_macro_precision_list = []
    all_macro_recall_list = []
    all_macro_auc_list = []
    all_macro_aupr_list = []
    all_micro_precision_list = []
    all_micro_recall_list = []
    all_micro_auc_list = []
    all_micro_aupr_list = []
    start_t = time()
    # train model
    # use 10-fold cross validation
    kf = KFold(n_splits=5, shuffle=True)
    for idx, (train_index, test_index) in enumerate(kf.split(masking, y)):
        # if idx > 0:
        #     break
        folder = idx + 1
        print(f"***********fold-{folder}***********")
        c_x_train = features[train_index]
        c_y_train = y.iloc[train_index].values

        c_x_test = features[test_index]
        c_y_test = y.iloc[test_index].values

        # c_x_test = c_x_test[:, np.newaxis]
        # c_x_train = c_x_train[:, np.newaxis]

        # Data.TensorDataset()
        train_x = torch.from_numpy(c_x_train)
        train_y = torch.from_numpy(c_y_train)
        test_x = torch.from_numpy(c_x_test)
        test_y = torch.from_numpy(c_y_test)
        train_fpt = torch.from_numpy(data_fp[train_index])
        test_fpt = torch.from_numpy(data_fp[test_index])
        wavelet_fp = torch.FloatTensor(wavelet_fp)
        train_wavelet = wavelet_fp[train_index]
        test_wavelet = wavelet_fp[test_index]
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

        loader = data.DataLoader(
            dataset=torch_dataset,
            batch_size=128, 
            shuffle=True,  
            num_workers=1  
        )

        # construct model & optimizer
        # model = GCNModel(args, data.n_entities, data.n_relations, entity_pre_embed, relation_pre_embed,
        #                  structure_pre_embed)

        # model = nn.Sequential(nn.Conv1d(in_channels=1, out_channels=4, kernel_size=2, padding="same"),
        #               # nn.MaxPool1d(2),
        #               nn.Flatten(),
        #               nn.BatchNorm1d(4 * 4 * 300),
        #               nn.Linear(4 * 4 * 300, 256),
        #               nn.LeakyReLU(args.alpha),
        #               # nn.Dropout(args.dropout),
        #               # nn.BatchNorm1d(1024),
        #               nn.Linear(256, 64),
        #               nn.LeakyReLU(args.alpha),
        #               # nn.Dropout(args.dropout),
        #               nn.Linear(64, 27),
        #               nn.Sigmoid())
        model = ADRModel(device)
        model.init_weights()
        model.to(device)
        # logging.info(model)

        # define optimizer and loss function
        optimizer = optim.Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
        if args.multi_type:
            loss_func = torch.nn.CrossEntropyLoss()
        else:
            # loss_func = torch.nn.MultiLabelSoftMarginLoss(reduction='mean')
            loss_func = torch.nn.BCEWithLogitsLoss()

        macro_precision_list = []
        macro_recall_list = []
        macro_auc_list = []
        macro_aupr_list = []

        micro_precision_list = []
        micro_recall_list = []
        micro_auc_list = []
        micro_aupr_list = []

        bad_counter = 0
        best_auc = 0
        best_epoch = 0

        time0 = time()
        for epoch in range(1, args.n_epoch + 1):
            out_list = np.empty((0, 27))
            loss_sum = 0
            batch_size = args.DDI_batch_size
            batch = math.ceil(c_y_train.shape[0] / batch_size)
            for batch_id in range(batch):
                batch_x, batch_fpt, batch_y = torch_dataset[batch_id*batch_size : (batch_id+1)*batch_size]
                model.train()
                # time1 = time()
                out,transfor_f = model(batch_x, batch_fpt, train_wavelet)
                print(transfor_f,batch_y.index)

                if not args.multi_type:
                    loss = loss_func(out, batch_y.float())
                else:
                    loss = loss_func(out, batch_y.long())

                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
                out_list = np.concatenate((out_list, torch.sigmoid(out).cpu().detach().numpy()), axis=0)
                loss_sum += loss.item()

            # train_acc, train_precision, train_recall, train_aupr, train_auc = calc_metrics(train_y, out, args.multi_type)
            train_auc = roc_auc_score(train_y.cpu().detach().numpy(), out_list, average='macro')
            logging.info('DDI Training: Folder {:04d} | Epoch {:04d} | AUC {:.4f} | Loss {:.4f}'.format(folder, epoch, train_auc, loss_sum))
            # logging.info(
            #     'DDI Training: Folder {:04d} | Epoch {:04d} | Precision {:.4f} Recall {:.4f} AUPR {:.4f} ACC '
            #     '{:.4f} AUC {:.4f} | Loss {:.4f}'.format(folder, epoch, train_precision, train_recall, train_aupr,
            #                                              train_acc, train_auc, loss.item()))

            bad_counter, best_auc, best_epoch = early_stopping(model, epoch, best_epoch, train_auc, best_auc, bad_counter)
            if bad_counter >= args.stopping_steps or epoch == args.n_epoch:
                model = load_model(model, args.save_dir, best_epoch)
                precision, recall, aupr, acc, auc, loss_list = evaluate(args, model, test_x, test_fpt, test_wavelet, test_y, test_index)
                logging.info(
                    'Final DDI Evaluation: Best_epoch {:04d} | Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} AUPR {:.4f} ACC '
                    '{:.4f} AUC {:.4f}'.format(best_epoch, time() - time0, precision, recall, aupr, acc, auc))
                all_precision_list.append(precision)
                all_recall_list.append(recall)
                all_aupr_list.append(aupr)
                all_acc_list.append(acc)
                all_auc_list.append(auc)
                all_loss_list = np.append(all_loss_list, loss_list, axis=0)
                break

            if not args.multi_type:
                pass
                # if (epoch % args.evaluate_every) == 0:
                #     precision, recall, aupr, acc, auc = evaluate(args, model, test_x, test_fpt, test_y)
                #     logging.info(
                #         'DDI Evaluation: Folder {:04d} | Epoch {:04d} | Precision {:.4f} Recall {:.4f} AUPR {:.4f} ACC '
                #         '{:.4f} AUC {:.4f}'.format(folder, epoch, precision, recall, aupr, acc, auc))

            else:
                pass
                # if (epoch % args.evaluate_every) == 0:
                #     time1 = time()
                #     macro_precision, macro_recall, macro_auc, macro_aupr, micro_precision, micro_recall, micro_auc, micro_aupr, acc, all_embedding = evaluate(
                #         args,
                #         model,
                #         test_x,
                #         test_y)
                #     logging.info(
                #         'DDI Evaluation: Epoch {:04d} | Total Time {:.1f}s | Macro Precision {:.4f} Macro Recall {:.4f} '
                #         'Macro AUC {:.4f} Macro AUPR {:.4f} Micro Precision {:.4f} Micro Recall {:.4f} Micro AUC {:.4f} Micro AUPR {:.4f} ACC {:.4f}'.format(
                #             epoch, time() - time1, macro_precision, macro_recall, macro_auc, macro_aupr, micro_precision,
                #             micro_recall,
                #             micro_auc, micro_aupr, acc))
                #
                #     epoch_list.append(epoch)
                #
                #     macro_precision_list.append(macro_precision)
                #     macro_recall_list.append(macro_recall)
                #     macro_auc_list.append(macro_auc)
                #     macro_aupr_list.append(macro_aupr)
                #
                #     micro_precision_list.append(micro_precision)
                #     micro_recall_list.append(micro_recall)
                #     micro_auc_list.append(micro_auc)
                #     micro_aupr_list.append(micro_aupr)
                #
                #     acc_list.append(acc)
                #     # auc_list.append(auc)
                #     best_acc, should_stop = early_stopping(acc_list, args.stopping_steps)
                #
                #     if should_stop:
                #         index = acc_list.index(best_acc)
                #         all_acc_list.append(acc_list[index])
                #         # all_auc_list.append(auc_list[index])
                #
                #         all_macro_precision_list.append(macro_precision_list[index])
                #         all_macro_recall_list.append(macro_recall_list[index])
                #         all_macro_auc_list.append(macro_auc_list[index])
                #         all_macro_aupr_list.append(macro_aupr_list[index])
                #
                #         all_micro_precision_list.append(micro_precision_list[index])
                #         all_micro_recall_list.append(micro_recall_list[index])
                #         all_micro_auc_list.append(micro_auc_list[index])
                #         all_micro_aupr_list.append(micro_aupr_list[index])
                #
                #         logging.info('Final DDI Evaluation: Macro Precision {:.4f} Macro Recall {:.4f} '
                #             'Macro AUC {:.4f} Macro AUPR {:.4f} Micro Precision {:.4f} Micro Recall {:.4f} Micro AUC {:.4f} Micro AUPR {:.4f} ACC {:.4f}'.format(
                #                 macro_precision, macro_recall, macro_auc, macro_aupr, micro_precision, micro_recall, micro_auc, micro_aupr, acc))
                #         break
                #
                #     if acc_list.index(best_acc) == len(acc_list) - 1:
                #         save_model(all_embedding, model, args.save_dir, epoch, best_epoch)
                #         logging.info('Save model on epoch {:04d}!'.format(epoch))
                #         best_epoch = epoch
                #
                #     if epoch == args.n_epoch:
                #         index = acc_list.index(best_acc)
                #         all_acc_list.append(acc_list[index])
                #         # all_auc_list.append(auc_list[index])
                #
                #         all_macro_precision_list.append(macro_precision_list[index])
                #         all_macro_recall_list.append(macro_recall_list[index])
                #         all_macro_auc_list.append(macro_auc_list[index])
                #         all_macro_aupr_list.append(macro_aupr_list[index])
                #
                #         all_micro_precision_list.append(micro_precision_list[index])
                #         all_micro_recall_list.append(micro_recall_list[index])
                #         all_micro_auc_list.append(micro_auc_list[index])
                #         all_micro_aupr_list.append(micro_aupr_list[index])
                #
                #         logging.info('Final DDI Evaluation: Macro Precision {:.4f} Macro Recall {:.4f} '
                #                      'Macro AUC {:.4f} Macro AUPR {:.4f} Micro Precision {:.4f} Micro Recall {:.4f} Micro AUC {:.4f} Micro AUPR {:.4f} ACC {:.4f}'.format(
                #             macro_precision, macro_recall, macro_auc, macro_aupr, micro_precision, micro_recall, micro_auc, micro_aupr, acc))
    if not args.multi_type:

        print(all_precision_list)
        print(all_recall_list)
        print(all_aupr_list)
        print(all_acc_list)
        print(all_auc_list)
        mean_acc = np.mean(all_acc_list)
        mean_precision = np.mean(all_precision_list)
        mean_recall = np.mean(all_recall_list)
        mean_aupr = np.mean(all_aupr_list)
        mean_auc = np.mean(all_auc_list)
        all_loss = np.array(all_loss_list)[1:].tolist()
        all_loss.sort(key=lambda item: item[0], reverse=False)
        np.savetxt(args.data_dir+'all_results.txt', all_loss)
        logging.info('10-fold cross validation DDI Mean Evaluation: Total Time {:.1f}s | Precision {:.4f} Recall {:.4f} AUPR {:.4f} ACC '
                     '{:.4f} AUC {:.4f}'.format(time()-start_t, mean_precision, mean_recall, mean_aupr, mean_acc, mean_auc))
    else:

        print(all_acc_list)
        print(all_macro_precision_list)
        print(all_macro_recall_list)
        print(all_macro_auc_list)
        print(all_macro_aupr_list)
        print(all_micro_precision_list)
        print(all_micro_recall_list)
        print(all_micro_auc_list)
        print(all_micro_aupr_list)

        mean_acc = np.mean(all_acc_list)
        mean_macro_precision = np.mean(all_macro_precision_list)
        mean_macro_recall = np.mean(all_macro_recall_list)
        mean_macro_auc = np.mean(all_macro_auc_list)
        mean_macro_aupr = np.mean(all_macro_aupr_list)
        mean_micro_precision = np.mean(all_micro_precision_list)
        mean_micro_recall = np.mean(all_micro_recall_list)
        mean_micro_auc = np.mean(all_micro_auc_list)
        mean_micro_aupr = np.mean(all_micro_aupr_list)
        # mean_auc = np.mean(all_auc_list)
        logging.info(
            '5-fold cross validation DDI Mean Evaluation: Macro Precision {:.4f} Macro Recall {:.4f} Macro AUC {:.4f} Macro AUPR {:.4f} Micro Precision {:.4f} Micro Recall {:.4f} Micro AUC {:.4f} Macro AUPR {:.4f} ACC {:.4f}'.format(
                mean_macro_precision, mean_macro_recall, mean_macro_auc, mean_macro_aupr, mean_micro_precision,
                mean_micro_recall,
                mean_micro_auc, mean_micro_aupr, mean_acc))


if __name__ == '__main__':
    args = parse_args()
    train(args)
