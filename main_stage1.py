# This file is for stage 1
from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import logging
import time
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, Predictor, Predictor_deep
from utils.utils import weights_init, set_logger
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset
from loaders.data_list import return_classlist


# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=50000, metavar='N',
                    help='maximum number of iterations to train (default: 50000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda_stage1',
                    help='dir to save checkpoint')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--log_interval', type=int, default=100, metavar='N',
                    help='how many batches to wait before logging training status')
parser.add_argument('--save_interval', type=int, default=1000, metavar='N',
                    help='how many batches to wait before saving a model')
parser.add_argument('--net', type=str, default='alexnet', choices=['alexnet', 'resnet34'],
                    help='which network to use')
parser.add_argument('--source', type=str, default='real',
                    help='source domain')
parser.add_argument('--target', type=str, default='sketch',
                    help='target domain')
parser.add_argument('--dataset', type=str, default='multi', choices=['multi', 'office_home'],
                    help='the name of dataset')
parser.add_argument('--num', type=int, default=3,
                    help='number of labeled examples in the target')
parser.add_argument('--patience', type=int, default=5, metavar='S',
                    help='early stopping to wait for improvment before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')
parser.add_argument('--exp_variation', type=str, default='s1_s+t', choices=['s1_s+t'],
                    help='experiment variations')
parser.add_argument('--num_workers', default=3, type=int, help='change num workers for debugging')
args = parser.parse_args()


# dataloader
source_loader, target_loader, target_loader_unl, target_loader_val, target_loader_test = return_dataset(args)

# path and record
record_dir = 'record/%s/%s' % (args.dataset, args.exp_variation)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir, '%s_%s_to_%s_num_%s_%s' %
                           (args.net, args.source, args.target, args.num, args.exp_variation))
set_logger(record_file + '.log')
if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)
root = './data/%s/' % args.dataset
image_set_file_s = os.path.join('./data/txt/%s' % args.dataset, 'labeled_source_images_' + args.source + '.txt')
image_set_file_t = os.path.join('./data/txt/%s' % args.dataset, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num))
image_set_file_unl = os.path.join('./data/txt/%s' % args.dataset, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))
class_list, source_data_num = return_classlist(image_set_file_s)
logging.info('\n\n')
logging.info('Start the experiment!')
logging.info('Dataset %s Source %s Target %s Labeled num per class %s Network %s' %(args.dataset, args.source, args.target, args.num, args.net))
logging.info("%d classses in this dataset" % len(class_list))


# manual seed
torch.cuda.manual_seed(args.seed)

# model
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
    label_batch_size = 48
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
    label_batch_size = 64
else:
    raise ValueError('Model cannot be recognized.')

# params for optimizer
params = []
for key, value in dict(G.named_parameters()).items():
    if value.requires_grad:
        if 'classifier' not in key:
            params += [{'params': [value], 'lr': args.multi, 'weight_decay': 0.0005}]
        else:
            params += [{'params': [value], 'lr': args.multi * 10, 'weight_decay': 0.0005}]

# Classifier
if "resnet" in args.net:
    F1 = Predictor_deep(num_class=len(class_list), inc=inc)
else:
    F1 = Predictor(num_class=len(class_list), inc=inc, temp=args.T)
weights_init(F1)

G.cuda()
F1.cuda()

# make Tensor
im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
im_data_train = torch.FloatTensor(1)
gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
gt_labels_train = torch.LongTensor(1)

# .cuda()
im_data_s = im_data_s.cuda()
im_data_t = im_data_t.cuda()
im_data_tu = im_data_tu.cuda()
im_data_train = im_data_train.cuda()
gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()
gt_labels_train = gt_labels_train.cuda()

# Variable
im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
im_data_train = Variable(im_data_train)
gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
gt_labels_train = Variable(gt_labels_train)


# train function start =================================================================================================
def train():

    global G
    global F1

    # will be used for stage 2
    full_save_check_path = os.path.join(args.checkpath, "model_{}_{}_to_{}_num_{}_{}.pth.tar".
                                        format(args.net, args.source, args.target, args.num, args.exp_variation))
    G.train()
    F1.train()

    # optimizer
    optimizer_g = optim.SGD(params, momentum=0.9, weight_decay=0.0005, nesterov=True)
    optimizer_f = optim.SGD(list(F1.parameters()), lr=1.0, momentum=0.9, weight_decay=0.0005, nesterov=True)
    def zero_grad_all():
        optimizer_g.zero_grad()
        optimizer_f.zero_grad()
    param_lr_g = []
    for param_group in optimizer_g.param_groups:
        param_lr_g.append(param_group["lr"])
    param_lr_f = []
    for param_group in optimizer_f.param_groups:
        param_lr_f.append(param_group["lr"])

    # loss
    criterion = nn.CrossEntropyLoss().cuda()

    all_step = args.steps

    # data_loader
    data_iter_s = iter(source_loader)
    data_iter_t = iter(target_loader)
    data_iter_tu = iter(target_loader_unl)
    len_train_source = len(source_loader)
    len_train_target = len(target_loader)
    len_train_target_unl = len(target_loader_unl)

    best_acc = 0
    counter = 0

    # step start =======================================================================================================
    for step in range(all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']

        # data from loader
        if step % len_train_source == 0:
            data_iter_s = iter(source_loader)
        if step % len_train_target == 0:
            data_iter_t = iter(target_loader)
        if step % len_train_target_unl == 0:
            data_iter_tu = iter(target_loader_unl)
        data_s = next(data_iter_s)
        data_t = next(data_iter_t)
        data_tu = next(data_iter_tu)

        with torch.no_grad():
            im_data_s.resize_(data_s[0].size()).copy_(data_s[0])
            im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
            im_data_tu.resize_(data_tu[0].size()).copy_(data_tu[0])
            gt_labels_s.resize_(data_s[1].size()).copy_(data_s[1])
            gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])

        data = torch.cat((im_data_s, im_data_t), 0)
        target = torch.cat((gt_labels_s, gt_labels_t), 0)

        if args.num == 0:
            data = im_data_s
            target = gt_labels_s

        zero_grad_all()

        # forward
        feature = G(data)
        output = F1(feature)
        loss = criterion(output, target)

        # backward
        loss.backward(retain_graph=True)
        optimizer_g.step()
        optimizer_f.step()

        zero_grad_all()

        # for logging
        log_train = 'S {} T {} Train Ep: {} lr{} \t Loss Classification: {:.6f} Method {}\n'. \
            format(args.source, args.target, step, lr, loss.data, args.exp_variation)

        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()

        if step % args.log_interval == 0:
            logging.info(log_train)

        # validation and test
        if step % args.save_interval == 0 and step > 0:
            logging.info('using test-set')
            loss_test, acc_test = test(target_loader_test)
            logging.info('using validation set')
            loss_val, acc_val = test(target_loader_val)
            G.train()
            F1.train()
            if acc_val >= best_acc:
                best_acc = acc_val
                best_acc_test = acc_test
                counter = 0
            else:
                counter += 1
            if args.early:
                if counter > args.patience:
                    # saving the best validation model
                    if args.save_check:
                        logging.info('saving the model of stage 1...')
                        torch.save({
                            'G': G.state_dict(),
                            'F1': F1.state_dict(),
                            'optimizer_g': optimizer_g.state_dict(),
                            'optimizer_f': optimizer_f.state_dict(),
                            'step': step + 1,
                        }, full_save_check_path)
                    break

            logging.info('best acc test %f best acc val %f' % (best_acc_test, acc_val))
            logging.info('record %s' % record_file)

            with open(record_file, 'a') as f:
                if step == args.save_interval:
                    f.write('Start the new experiment in stage 1 ========== ')
                    now = time.localtime()
                    f.write("%04d/%02d/%02d %02d:%02d\n" % (
                    now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min))
                f.write('step %d best %f final %f \n' % (step, best_acc_test, acc_val))
                logging.info('step %d best test %f final validation %f \n' % (step, best_acc_test, acc_val))
            G.train()
            F1.train()


def test(loader):
    global im_data_t
    global gt_labels_t

    G.eval()
    F1.eval()
    test_loss = 0
    correct = 0
    size = 0
    num_class = len(class_list)
    output_all = np.zeros((0, num_class))
    criterion = nn.CrossEntropyLoss().cuda()
    confusion_matrix = torch.zeros(num_class, num_class)
    with torch.no_grad():
        for batch_idx, data_t in enumerate(loader):
            with torch.no_grad():
                im_data_t.resize_(data_t[0].size()).copy_(data_t[0])
                gt_labels_t.resize_(data_t[1].size()).copy_(data_t[1])
            feat = G(im_data_t)
            output1 = F1(feat)
            output_all = np.r_[output_all, output1.data.cpu().numpy()]
            size += im_data_t.size(0)
            pred1 = output1.data.max(1)[1]
            for t, p in zip(gt_labels_t.view(-1), pred1.view(-1)):
                confusion_matrix[t.long(), p.long()] += 1
            correct += pred1.eq(gt_labels_t.data).cpu().sum()
            test_loss += criterion(output1, gt_labels_t) / len(loader)
    logging.info('Test set: Average loss: {:.4f}, Accuracy: {}/{} F1 ({:.0f}%)'.format(test_loss, correct, size, 100. * correct / size))
    return test_loss.data, 100. * float(correct) / size


train()
