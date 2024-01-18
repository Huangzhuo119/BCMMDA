# !/usr/bin/env python
# -*- coding:utf-8 -*-
import os, datetime, sys, copy
# sys.path.append("../..")
import torch.utils.data
import csv
from torch.utils.data import DataLoader
from torch.autograd import Variable
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve,f1_score, roc_curve, confusion_matrix, \
    precision_score, recall_score, auc
from BCMMDA_dataset import *
from mymodel import *
from BCMMDA_config import *
from BCMMDA_utils import *
from copy import deepcopy
from lifelines.utils import concordance_index
from sklearn.model_selection import KFold, StratifiedKFold




if torch.cuda.is_available():
    device = torch.device("cuda:0")  # "cuda:0"
else:
    device = torch.device("cpu")


def test(data_generator, model):
    y_pred = []
    y_label = []
    fwhat=[]
    score1=[]
    proteins=[]
    drugs=[]
    model.eval()
    loss_accumulate = 0.0
    count = 0.0
    for i, (d, f, p, label) in enumerate(data_generator):
        score = model(d.long().to(device), p.long().to(device))

        logits = torch.squeeze(score)
        loss_fct = torch.nn.BCELoss()
        np.seterr(divide='ignore', invalid='ignore')
        label = Variable(torch.from_numpy(np.array(label)).float()).to(device)

        loss = loss_fct(logits, label)

        loss_accumulate += loss
        count += 1

        logits = logits.detach().cpu().numpy()

        label_ids = label.to('cpu').numpy()
        protein_ids = p.long().to('cpu').numpy()
        # drug_ids = d.long().to('cpu').numpy()
        score_ids = score.to('cpu').numpy()

        # proteins = proteins + protein_ids.tolist()
        # drugs = drugs + drug_ids.tolist()
        score1=score1+ score_ids.tolist()

        y_label = y_label + label_ids.flatten().tolist()
        y_pred = y_pred + logits.flatten().tolist()

    loss = loss_accumulate / count

    fpr, tpr, thresholds = roc_curve(y_label, y_pred)

    precision1 = tpr / (tpr + fpr)

    f1 = 2 * precision1 * tpr / (tpr + precision1 + 0.00001)

    thred_optim = thresholds[5:][np.argmax(f1[5:])]

    # print("optimal threshold: " + str(thred_optim))

    y_pred_s = [1 if i else 0 for i in (y_pred >= thred_optim)]

    auc_k = auc(fpr, tpr)
    # print("AUROC:" + str(auc_k))
    # print("AUPRC: " + str(average_precision_score(y_label, y_pred)))
    precisiont, recallt, _ = precision_recall_curve(y_label, score1)
    cm1 = confusion_matrix(y_label, y_pred_s)
    recall = recall_score(y_label, y_pred_s)
    precision = precision_score(y_label, y_pred_s)



    total1 = sum(sum(cm1))
    #####from confusion matrix calculate accuracy
    accuracy1 = (cm1[0, 0] + cm1[1, 1]) / total1
    #print('Accuracy : ', accuracy1)

    sensitivity1 = cm1[0, 0] / (cm1[0, 0] + cm1[0, 1])
    # print('Sensitivity : ', sensitivity1)

    specificity1 = cm1[1, 1] / (cm1[1, 0] + cm1[1, 1])
    # print('Specificity : ', specificity1)

    outputs = np.asarray([1 if i else 0 for i in (np.asarray(y_pred) >= 0.5)])
    # return roc_auc_score(y_label, y_pred), average_precision_score(y_label,
    #                                                                y_pred), accuracy1,precision, recall, f1_score(
    #     y_label,
    #     outputs), y_pred, loss.item()

    # 保存为txt文件
    # 将y_label和y_pred保存为列表
    y_label_list = [str(item) for item in y_label]
    y_preds_list = [str(item) for item in y_pred_s]
    score1_list = [str(item) for item in score1]
    proteins_list = [str(item) for item in proteins]
    drugs_list = [str(item) for item in drugs]
    fwhat_list = [str(item) for item in fwhat]


    # # 保存y_label列表为txt文件
    # with open('y_label.txt', 'w') as f:
    #     for item in y_label_list:
    #         f.write("%s\n" % item)
    # with open('score1.txt', 'w') as f:
    #     for item in score1_list:
    #         f.write("%s\n" % item)
    # # 保存y_pred列表为txt文件
    # with open('y_preds.txt', 'w') as f:
    #     for item in y_preds_list:
    #         f.write("%s\n" % item)





    return roc_auc_score(y_label, y_pred), accuracy1, precision, recall,auc(recallt, precisiont), logits, loss.item()

def formula_output(prefix, epoch, auc, acc,p, r,aupr,loss):
    return dataset_name + " " + prefix + " at" + " Epoch " + epoch + " AUC: " + auc +" ACC: "+acc+ " Precision: " + p + " Recall: " + r + " AUPR: " + aupr + " Loss: " + loss



def train(config):
    loss_history = []
    best_model = 0
    best_accuracy = 0
    counter = 0
    model = SSCNN_DTI(config)

    model = model.to(device)

    opt = torch.optim.Adam(model.parameters(), lr=config['lr'])
    max_auc = 0
    model_max = deepcopy(model)

    with torch.set_grad_enabled(False):
        auc,  acc, p, recall, aupr, logits, loss = test(config['test_loader'], model_max)
        print(formula_output("Testing", "-1", str(auc),  str(acc), str(p), str(recall), str(aupr), str(loss)))


    print('--- Go for Training ---')
    torch.backends.cudnn.benchmark = True
    for epo in range(config['epochs']):
        model.train()

        for i, (d, f, p, label) in enumerate(train_loader):
            score = model(d.long().to(device), p.long().to(device))
            np.seterr(divide='ignore', invalid='ignore')
            label = Variable(torch.from_numpy(np.array(label)).float()).to(device)

            loss_fct = torch.nn.BCELoss()
            # m = torch.nn.Sigmoid()
            n = torch.squeeze(score)

            loss = loss_fct(n, label)
            loss_history.append(loss)

            opt.zero_grad()
            loss.backward()
            if config['clip']:
                torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
            opt.step()

            if (i % 100
                    + 0 == 0):
                # print(formula_output("Training", str(epo + 1), str(auc), str(p), str(recall), str(f1), str(loss.cpu().detach().numpy())))
                print('Training at Epoch ' + str(epo + 1) + ' iteration ' + str(i) + ' with loss ' + str(
                    loss.cpu().detach().numpy()))


        # every epoch test
        with torch.set_grad_enabled(False):
            auc,  acc, p, recall, aupr, logits, loss = test(valid_loader, model)
            # auc, auprc, acc, p, recall, f1, logits, loss = test(valid_loader, model)
            valid_result_best.append([dataset_name, str(epo), str(auc), str(p), str(recall)])
            if acc > best_accuracy and epo >= 3:
                best_accuracy = acc
                counter = 0
                model_max = deepcopy(model)
                save_dir = './model/' + dataset_name + "/"
                assert_dir_exist(save_dir)
                print("new model saved")

                save_best_model(model, model_dir=save_dir, best_epoch=i)
            else:
                counter += 1
            if counter == args['stop_counter']:
                print("early stop at epoch %d" % epo)
                break
            print(
                formula_output("Validation", str(epo + 1), str(auc),  str(acc), str(p), str(recall), str(aupr),
                               str(loss)))
            # print(formula_output("Validation", str(epo + 1), str(auc), str(auprc),str(acc), str(p), str(recall), str(f1),
            #                      str(loss)))
            # print('Validation at Epoch ' + str(epo + 1) + ' , AUC: ' + str(auc) + ' , AUPRC: ' + str(
            #     auprc)+ ' , ACC: ' + str(
            #     acc)+ ' , Precision: ' + str(
            #     p)+ ' , Recall: ' + str(
            #     recall) + ' , F1: ' + str(f1))




    print('--- Go for Testing ---')
    try:
        with torch.set_grad_enabled(False):
            auc, acc, p, recall, aupr, logits, loss = test(test_loader, model_max)
            # add_to_xlxs(k,cnn_block_num,auc)

            print(formula_output("Testing", " ", str(auc),  str(acc), str(p), str(recall), str(aupr),
                                 str(loss)))
            # auc, auprc, acc, p, recall, f1, logits, loss = test(test_loader, model_max)
            # # add_to_xlxs(k,cnn_block_num,auc)
            # print(formula_output("Testing", " ", str(auc), str(auprc),str(acc), str(p), str(recall), str(f1),
            #                      str(loss)))
            test_result.append([dataset_name, str(auc),str(acc), str(p), str(recall),str(aupr)])

    except Exception as e:
        # print(e)
        print(e, 'testing failed')

    return model_max, loss_history


if __name__ == "__main__":
    valid_result_best = []
    test_result = []

    best_output = []
    # last_auc=best_model

    # dataset_name = 'BindingDB'
    # print(dataset_config)

    # hpyer parameter
    # k,1,2,3,4,5,6
    # n,1,2,3,4,5,6
    # k = int(sys.argv[1])
    # n = int(sys.argv[2])
    # dataset_name = sys.argv[3]

    k = 3
    n = 3
    dataset_name = "human"
    split_random = True
    # dataset_name=sys.argv[1]
    # split_random=bool(sys.argv[2])
    cnn_block_num = n
    dataset = dataset_config[dataset_name]
    root_path = dataset_config[dataset_name]

    input_path = root_path + "/"
    output_path = root_path + "/" + "output"

    decompose = "bcm"
    decompose_protein = "category"

    trainSmiles, trainProtein, trainLabel, \
    valSmiles, valProtein, valLabel, \
    testSmiles, testProtein, testLabel, \
    frag_set_d, frag_set_p, \
    frag_len_d, frag_len_p, \
    words2idx_d, words2idx_p = load_train_val_test_set(input_path, decompose,
                                                       decompose_protein="category", unseen_smiles=False, k=k,
                                                       split_random=split_random)
    args = SSCNN_args()



    print("positive ratio", sum(trainLabel + valLabel + testLabel) / len(trainLabel + valLabel + testLabel),
          len(trainLabel + valLabel + testLabel), sum(trainLabel + valLabel + testLabel))
    print("train:val:test", len(trainSmiles), len(valSmiles), len(testSmiles))


    args['max_drug_seq'] = max(frag_len_d)
    args['max_protein_seq'] = max(frag_len_p)

    args['input_d_dim'] = len(frag_set_d) + 1
    args['input_p_dim'] = len(frag_set_p) + 1

    print("input_d_dim:", args['input_d_dim'])
    print("input_p_dim:", args['input_p_dim'])

    args['d_channel_size'][n - 1][0] = args['max_drug_seq']
    args['p_channel_size'][n - 1][0] = args['max_protein_seq']

    args['d_channel_size'] = args['d_channel_size'][n - 1]
    args['p_channel_size'] = args['p_channel_size'][n - 1]

    trainDataset = NewDataset(trainSmiles, trainProtein, trainLabel, words2idx_d, words2idx_p, args['max_drug_seq'],
                              args['max_protein_seq'])
    validDataset = NewDataset(valSmiles, valProtein, valLabel, words2idx_d, words2idx_p, args['max_drug_seq'],
                              args['max_protein_seq'])

    testDataset = NewDataset(testSmiles, testProtein, testLabel, words2idx_d, words2idx_p, args['max_drug_seq'],
                             args['max_protein_seq'])

    dataFold = torch.utils.data.ConcatDataset([trainDataset, validDataset])

    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    for fold,(train_idx, valid_idx) in enumerate(kf.split(dataFold)) :

        train_fold = torch.utils.data.dataset.Subset(dataFold, train_idx)
        valid_fold = torch.utils.data.dataset.Subset(dataFold, valid_idx)

        train_loader = DataLoader(dataset=train_fold, batch_size=args['batch_size'], shuffle=True, drop_last=True)
        valid_loader = DataLoader(dataset=valid_fold, batch_size=args['batch_size'], shuffle=True, drop_last=True)

        test_loader = DataLoader(dataset=testDataset, batch_size=args['batch_size'], shuffle=True, drop_last=True)



    args['train_loader'] = train_loader
    args['valid_loader'] = valid_loader
    args['test_loader'] = test_loader

    model_max, _ = train(args)
    print(valid_result_best, test_result)
    if split_random:
        write2txt("./data/test_result" + "_split_random_" + dataset_name, test_result)
    else:
        write2txt("./data/test_result" + "_no_split_random_" + dataset_name, test_result)
