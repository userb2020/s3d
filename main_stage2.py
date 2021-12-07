# This file is for stage 2
from __future__ import print_function
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import math
import logging
import time
from torch.autograd import Variable
from model.resnet import resnet34
from model.basenet import AlexNetBase, Predictor, Predictor_deep
from utils.utils import weights_init, set_logger
from utils.lr_schedule import inv_lr_scheduler
from utils.return_dataset import return_dataset, return_stage2_dataset, calc_mean_margin, ResizeImage
from utils.loss import KDLoss
from loaders.data_list import return_classlist
from torchvision import transforms


# Training settings
parser = argparse.ArgumentParser(description='SSDA Classification')
parser.add_argument('--steps', type=int, default=100000, metavar='N',
                    help='maximum number of iterations to train (default: 50000)')
parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                    help='learning rate (default: 0.001)')
parser.add_argument('--multi', type=float, default=0.1, metavar='MLT',
                    help='learning rate multiplication')
parser.add_argument('--T', type=float, default=0.05, metavar='T',
                    help='temperature (default: 0.05)')
parser.add_argument('--save_check', action='store_true', default=False,
                    help='save checkpoint or not')
parser.add_argument('--checkpath', type=str, default='./save_model_ssda_stage2',
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
                    help='early stopping to wait for improvement before terminating. (default: 5 (5000 iterations))')
parser.add_argument('--early', action='store_false', default=True,
                    help='early stopping on validation or not')
parser.add_argument('--num_workers', default=3, type=int, help='change num workers for debugging')
# for pair distillation
parser.add_argument('--temp', default=4.0, type=float, help='temperature scaling')
# experiment variation for stage 1 and 2
parser.add_argument('--s1_exp_variation', default='s1_s+t', type=str)
parser.add_argument('--exp_variation', default='s2_s3d', type=str, help='experiment variation')
# pseudo label update cycle
parser.add_argument('--pseudo_interval', default=100, type=int)
# kd_loss weight factor lamb
parser.add_argument('--kd_lambda', default=8, type=float) # if set to 100, there will be no weight on kd_loss
# beta distribution parameter
parser.add_argument('--sty_w', default=0.1, type=float)
# layer2 means assistant generation module is applied until layer2
parser.add_argument('--sty_layer', default= 'layer1', type=str, choices=['layer1', 'layer2', 'layer3', 'layer4'])
parser.add_argument('--alpha_value', default=0.95, type=float)
args = parser.parse_args()


# seed
np.random.seed(args.seed) # fixing numpy operations
torch.manual_seed(args.seed) # fixing cpu operations
torch.cuda.manual_seed(args.seed) # fixing gpu operations
torch.cuda.manual_seed_all(args.seed) # fixing multi gpu operations
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


def adaptation_factor(x, kd_lambda):
    if x >= 1.0:
        return 1.0
    den = 1.0 + math.exp(-kd_lambda * x)
    lamb = 2.0 / den - 1.0
    return lamb


def cal_start_step_for_weight(kd_lambda, stage1_acc):
    # get a lambda step <- not a real step but for getting a weight
    v_acc = stage1_acc * 1.5
    v_acc = v_acc / 100
    if v_acc >= 1.0:
        return 100000
    lambda_step = - (50000 / kd_lambda) * math.log(2/(1 + v_acc) - 1)
    lambda_step = int(lambda_step)
    return lambda_step


# dataloader for test
_, _, _, target_loader_val, target_loader_test = return_dataset(args)
# path and record
record_dir = 'record/%s/%s' % (args.dataset, args.exp_variation)
if not os.path.exists(record_dir):
    os.makedirs(record_dir)
record_file = os.path.join(record_dir, '%s_%s_to_%s_num_%s_%s_lamb_%s' %
                           (args.net, args.source, args.target, args.num, args.exp_variation,
                            args.kd_lambda))
set_logger(record_file + '.log')
if os.path.exists(args.checkpath) == False:
    os.mkdir(args.checkpath)

image_set_file_s = os.path.join('./data/txt/%s' % args.dataset, 'labeled_source_images_' + args.source + '.txt')
image_set_file_t = os.path.join('./data/txt/%s' % args.dataset, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num))
image_set_file_unl = os.path.join('./data/txt/%s' % args.dataset, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))
if args.num == 0:
    image_set_file_t = os.path.join('./data/txt/%s' % args.dataset,'labeled_target_images_' + args.target + '_3.txt') # This is a dummy file for uda exp
    image_set_file_unl = os.path.join('./data/txt/%s' % args.dataset, 'labeled_source_images_' + args.target + '.txt')
class_list, source_data_num = return_classlist(image_set_file_s)
_, target_data_num = return_classlist(image_set_file_t)
lab_data_num = source_data_num + target_data_num

logging.info('\n\n')
logging.info('Start the experiment in stage 2!')
logging.info('Dataset %s Source %s Target %s Labeled num per class %s Network %s' %(args.dataset, args.source, args.target, args.num, args.net))
logging.info("%d classes in this dataset" % len(class_list))

# model
if args.net == 'resnet34':
    G = resnet34()
    inc = 512
    label_batch_size = 48
    crop_size = 224
elif args.net == "alexnet":
    G = AlexNetBase()
    inc = 4096
    label_batch_size = 64
    crop_size = 227
else:
    raise ValueError('Model cannot be recognized.')

# data transforms
data_transforms = {
        'train': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'val': transforms.Compose([
            ResizeImage(256),
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }

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
#im_data_s = torch.FloatTensor(1)
im_data_t = torch.FloatTensor(1)
im_data_tu = torch.FloatTensor(1)
im_data_train = torch.FloatTensor(1)
#gt_labels_s = torch.LongTensor(1)
gt_labels_t = torch.LongTensor(1)
gt_labels_train = torch.LongTensor(1)
all_batch_indices = torch.LongTensor(1)

# .cuda()
#im_data_s = im_data_s.cuda()
im_data_t = im_data_t.cuda()
im_data_tu = im_data_tu.cuda()
im_data_train = im_data_train.cuda()
#gt_labels_s = gt_labels_s.cuda()
gt_labels_t = gt_labels_t.cuda()
gt_labels_train = gt_labels_train.cuda()
all_batch_indices = all_batch_indices.cuda()

# Variable
#im_data_s = Variable(im_data_s)
im_data_t = Variable(im_data_t)
im_data_tu = Variable(im_data_tu)
im_data_train = Variable(im_data_train)
#gt_labels_s = Variable(gt_labels_s)
gt_labels_t = Variable(gt_labels_t)
gt_labels_train = Variable(gt_labels_train)
all_batch_indices = Variable(all_batch_indices)


# train function start =================================================================================================
def train():
    # stage 1 model path
    full_save_check_path_stage1 = os.path.join("save_model_ssda_stage1", "model_{}_{}_to_{}_num_{}_{}.pth.tar".
                                        format(args.net, args.source, args.target, args.num, args.s1_exp_variation))
    # stage 2 model path
    full_save_check_path_stage2 = os.path.join("save_model_ssda_stage2", "model_{}_{}_to_{}_num_{}_{}.pth.tar".
                                               format(args.net, args.source, args.target, args.num,
                                                      args.exp_variation))
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

    # loss function
    logging.info('using cross entropy loss for labeled examples')
    criterion = nn.CrossEntropyLoss().cuda()
    unl_criterion = nn.CrossEntropyLoss(reduction='none').cuda()
    kdloss = KDLoss(args.temp).cuda()
    cos = nn.CosineSimilarity(dim=1, eps=1e-6).cuda()
    softmax = nn.Softmax(dim=1)

    all_step = args.steps

    best_acc = 0
    counter = 0
    start_step = 0

    # load stage1 model
    logging.info("loading stage 1 model from " + full_save_check_path_stage1)
    checkpoint = torch.load(full_save_check_path_stage1)
    G.load_state_dict(checkpoint['G'])
    F1.load_state_dict(checkpoint['F1'])
    optimizer_g.load_state_dict(checkpoint['optimizer_g'])
    optimizer_f.load_state_dict(checkpoint['optimizer_f'])
    start_step = checkpoint['step']

    # seed
    np.random.seed(args.seed)  # fixing numpy operations
    torch.manual_seed(args.seed)  # fixing cpu operations
    torch.cuda.manual_seed(args.seed)  # fixing gpu operations
    torch.cuda.manual_seed_all(args.seed)  # fixing multi gpu operations
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logging.info('test stage1 pre-trained model before entering stage2')
    stage1_loss_test, stage1_acc_test = test(target_loader_test)

    # lambda_step will be the first step when calculating weight for kd_loss
    first_lambda_step = cal_start_step_for_weight(args.kd_lambda, stage1_acc_test)

    # calculating mean margin
    root = './data/%s/' % args.dataset
    logging.info('calculating mean margin...')
    stage2_margin = calc_mean_margin(image_set_file_unl, G, F1, root, data_transforms, args)
    logging.info('\nmean margin became: ' + str(stage2_margin))
    torch.cuda.empty_cache()

    # step start =======================================================================================================
    for step in range(start_step, all_step):
        optimizer_g = inv_lr_scheduler(param_lr_g, optimizer_g, step, init_lr=args.lr)
        optimizer_f = inv_lr_scheduler(param_lr_f, optimizer_f, step, init_lr=args.lr)
        lr = optimizer_f.param_groups[0]['lr']

        lambda_step = first_lambda_step + (step - start_step)
        lamb = adaptation_factor(lambda_step * 1.0 / 50000, args.kd_lambda)

        # pseudo label update and return pair dataset
        interval = args.pseudo_interval
        if (step % interval == 0) or (step == start_step):
            # generate pseudo label file and return train loader
            train_loader = return_stage2_dataset(args, G, F1, logging, stage2_margin)
            data_iter_train = iter(train_loader)
        data_train = next(data_iter_train)

        # data and target
        with torch.no_grad():
            im_data_train.resize_(data_train[0].size()).copy_(data_train[0])
            gt_labels_train.resize_(data_train[1].size()).copy_(data_train[1])

        zero_grad_all()

        data = im_data_train
        target = gt_labels_train
        batch_size = data.size(0)
        lab_data = data[:batch_size // 2]  # the first part of the data is labeled data
        lab_target = target[:batch_size // 2]
        unl_data = data[batch_size // 2:]  # the second part of the data is unlabeled target data

        # cross-entropy loss for labeled samples
        lab_feature, x_sty = G.forward_mean_var(lab_data, args.sty_layer)
        lab_output = F1(lab_feature)
        lab_loss = criterion(lab_output, lab_target)

        # generating assistant features
        unl_feature, assistant_feature = G.forward_assistant(unl_data, x_sty, args.sty_w, args.sty_layer)
        unl_output = F1(unl_feature)
        assistant_output = F1(assistant_feature)

        # unlabeled target weighted cross-entropy loss
        with torch.no_grad():
            unl_pred = softmax(unl_output)
            max_unl_pred = torch.max(unl_pred, dim=1)[0]
            pseudo_label = torch.max(unl_pred, dim=1)[1]
        unl_loss = unl_criterion(unl_output, pseudo_label.detach())
        weighted_unl_loss = torch.mul(max_unl_pred.detach(), unl_loss)
        mean_unl_loss = torch.mean(weighted_unl_loss)

        # kd_loss
        kd_loss = kdloss(unl_output, assistant_output.detach())

        # sum all the loss
        loss = lab_loss + (lamb * kd_loss) + mean_unl_loss

        # backward
        loss.backward()
        optimizer_g.step()
        optimizer_f.step()

        G.zero_grad()
        F1.zero_grad()
        zero_grad_all()

        # logging =========================================================
        log_train = 'S {} T {} Train Ep: {} lr{} \t ' \
                    'Loss Classification: {:.6f} UNL Loss Classification: {:.6f} Loss KD: {:.6f} Loss total: {:.6f} Method {}' \
                    .format(args.source, args.target, step, lr, lab_loss.data, mean_unl_loss.data, kd_loss.data,
                    loss.data, args.exp_variation)

        if step % args.log_interval == 0:
            logging.info(log_train)

        # validation and test ==========================================================================================
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
                    if args.save_check:
                        logging.info('saving the model of stage 2...')
                        torch.save({
                            'G': G.state_dict(),
                            'F1': F1.state_dict(),
                            'optimizer_g': optimizer_g.state_dict(),
                            'optimizer_f': optimizer_f.state_dict(),
                            'step': step + 1,
                        }, full_save_check_path_stage2)
                        with open(record_file, 'a') as f:
                            f.write('End the experiment ========== ')
                            now = time.localtime()
                            f.write("%04d/%02d/%02d %02d:%02d\n" % (
                            now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min))
                        break

            logging.info('best acc test %f best acc val %f' % (best_acc_test, acc_val))
            logging.info('record %s' % record_file)
            with open(record_file, 'a') as f:
                if step == start_step - 1 + args.save_interval:
                    f.write('Start the new experiment ========== ')
                    #f.write('something to write' + '\n')
                    now = time.localtime()
                    f.write("%04d/%02d/%02d %02d:%02d\n" % (now.tm_year, now.tm_mon, now.tm_mday, now.tm_hour, now.tm_min))
                f.write('step %d best %f final %f \n' % (step, best_acc_test, acc_val))
                if args.num == 0:
                    f.write('step %d test %f final %f \n' % (step, acc_test, acc_val))
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
            # change for torch1
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
