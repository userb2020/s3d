import torch
import torchvision, random, os
import torch.nn as nn
import time
import datetime
from torchvision import transforms
from loaders.data_list import Imagelists_VISDA, Imagelists_concat3, Imagelists_concat2
from torch.utils.data import Dataset, Sampler
from collections import defaultdict
from utils.utils import printProgress
from loaders.data_list import return_classlist


class ResizeImage():
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (int(size), int(size))
        else:
            self.size = size

    def __call__(self, img):
        th, tw = self.size
        return img.resize((th, tw))


# variations of return_dataset function ================================================================================
def return_dataset(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = \
        os.path.join(base_path,
                     'labeled_source_images_' +
                     args.source + '.txt')
    if args.num == 0:
        image_set_file_t = \
            os.path.join(base_path,
                         'labeled_target_images_' +
                         args.target + '_3.txt')    # this file is dummy (for uda exp)
    else:
        image_set_file_t = \
            os.path.join(base_path,
                        'labeled_target_images_' +
                        args.target + '_%d.txt' % (args.num))
    image_set_file_t_val = \
        os.path.join(base_path,
                     'validation_target_images_' +
                     args.target + '_3.txt')
    image_set_file_unl = \
        os.path.join(base_path,
                     'unlabeled_target_images_' +
                     args.target + '_%d.txt' % (args.num))
    if args.num == 0: # for uda setting
        image_set_file_unl = \
            os.path.join(base_path,
                         'labeled_source_images_' +
                         args.target + '.txt')

    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224

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

    source_dataset = Imagelists_VISDA(image_set_file_s, root=root,
                                      transform=data_transforms['train'])
    target_dataset = Imagelists_VISDA(image_set_file_t, root=root,
                                      transform=data_transforms['val'])
    target_dataset_val = Imagelists_VISDA(image_set_file_t_val, root=root,
                                          transform=data_transforms['val'])
    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root,
                                          transform=data_transforms['val'])
    target_dataset_test = Imagelists_VISDA(image_set_file_unl, root=root,
                                           transform=data_transforms['test'])

    if args.net == 'alexnet':
        bs = 32
        source_bs = bs
        if args.num == 0:
            source_bs = 2 * bs
    else:
        bs = 24
        source_bs = bs
        if args.num == 0:
            source_bs = 2 * bs

    source_loader = torch.utils.data.DataLoader(source_dataset, batch_size=source_bs,
                                                num_workers=args.num_workers, shuffle=True,
                                                drop_last=True)
    target_loader = torch.utils.data.DataLoader(target_dataset,
                                    batch_size=min(bs, len(target_dataset)),
                                    num_workers=args.num_workers,
                                    shuffle=True, drop_last=True)
    target_loader_val = \
        torch.utils.data.DataLoader(target_dataset_val,
                                    batch_size=min(bs, len(target_dataset_val)),
                                    num_workers=args.num_workers,
                                    shuffle=True, drop_last=True)
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=args.num_workers,
                                    shuffle=True, drop_last=True)
    target_loader_test = \
        torch.utils.data.DataLoader(target_dataset_test,
                                    batch_size=bs * 2, num_workers=args.num_workers,
                                    shuffle=True, drop_last=True)


    return source_loader, target_loader, target_loader_unl, \
            target_loader_val, target_loader_test


def return_dataset_test(args):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, args.source + '_all' + '.txt')
    image_set_file_test = os.path.join(base_path,
                                       'unlabeled_target_images_' +
                                       args.target + '_%d.txt' % (args.num))
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
    data_transforms = {
        'test': transforms.Compose([
            ResizeImage(256),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ]),
    }
    target_dataset_unl = Imagelists_VISDA(image_set_file_test, root=root,
                                          transform=data_transforms['test'],
                                          test=True)
    class_list = return_classlist(image_set_file_s)
    print("%d classes in this dataset" % len(class_list))
    if args.net == 'alexnet':
        bs = 32
    else:
        bs = 24
    target_loader_unl = \
        torch.utils.data.DataLoader(target_dataset_unl,
                                    batch_size=bs * 2, num_workers=args.num_workers,
                                    shuffle=False, drop_last=False)
    return target_loader_unl, class_list


def return_stage2_dataset(args, G, F1, logging, stage2_margin):
    base_path = './data/txt/%s' % args.dataset
    root = './data/%s/' % args.dataset
    image_set_file_s = os.path.join(base_path, 'labeled_source_images_' + args.source + '.txt')
    if args.num == 0:
        image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_3.txt') # dummy file
        image_set_file_unl = os.path.join(base_path, 'labeled_source_images_' + args.target + '.txt')
    else:
        image_set_file_t = os.path.join(base_path, 'labeled_target_images_' + args.target + '_%d.txt' % (args.num))
        image_set_file_unl = os.path.join(base_path, 'unlabeled_target_images_' + args.target + '_%d.txt' % (args.num))

    pseudo_image_set_file_unl = os.path.join(base_path, 'pseudo_%s_%s' % (args.source, args.target),
                                                 'pseudo_unlabeled_target_images_' + args.target + '_%d_%s_%s.txt' % (
                                                     args.num, args.kd_lambda, args.sty_layer))
    # data transforms per each network
    if args.net == 'alexnet':
        crop_size = 227
    else:
        crop_size = 224
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
        ])
    }

    # pseudo label generation ========================================
    # generate pseudo_image_set_file_unl file
    start = time.time()

    logging.info('\nmaking pseudo label...')
    largest_margin, mean_margin, margin_list = make_pseudo_label_rss \
        (image_set_file_unl, pseudo_image_set_file_unl, G, F1, root, data_transforms, stage2_margin, args)

    sec = time.time() - start
    times = str(datetime.timedelta(seconds=sec)).split(".")
    times = times[0]
    logging.info('\n' + times + ' time takes ...')

    # train dataset that concat S, Tl and Tu
    class_list, _ = return_classlist(image_set_file_s)

    if args.num == 0:
        train_dataset = DatasetWrapper(Imagelists_concat2(image_set_file_s, pseudo_image_set_file_unl,
                                root=root, transform=data_transforms['train']))
    else:
        train_dataset = DatasetWrapper(Imagelists_concat3(image_set_file_s, image_set_file_t, pseudo_image_set_file_unl,
                                        root=root, transform=data_transforms['train']))
    if args.net == 'alexnet':
        bs = 64 # 64 for unlabeled target, 32 for labeled target, 32 for source
    elif args.net == 'resnet34':
        bs = 48
    if args.num == 0:
        train_loader = torch.utils.data.DataLoader(train_dataset,
                                                   batch_sampler=PairBatchSampler_for_uda(train_dataset,
                                                    2 * bs, num_iterations=args.pseudo_interval), num_workers=args.num_workers)
    else:
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=PairBatchSampler(train_dataset,
                                                              2 * bs, num_iterations=args.pseudo_interval),num_workers=args.num_workers)

    return train_loader


# make dictionary which key contains class and value contains sample indices ======
class DatasetWrapper(Dataset):
    # Additinoal attributes
    # - indices
    # - classwise_indices
    # - num_classes
    # - get_class

    def __init__(self, dataset, indices=None):
        self.base_dataset = dataset
        if indices is None:
            self.indices = list(range(len(dataset)))
        else:
            self.indices = indices
        '''
        # torchvision 0.2.0 compatibility
        if torchvision.__version__.startswith('0.2'):
            if isinstance(self.base_dataset, datasets.ImageFolder):
                self.base_dataset.targets = [s[1] for s in self.base_dataset.imgs]
            else:
                if self.base_dataset.train:
                    self.base_dataset.targets = self.base_dataset.train_labels
                else:
                    self.base_dataset.targets = self.base_dataset.test_labels
        '''

        self.classwise_indices = defaultdict(list)
        for i in range(len(self)):
            y = self.base_dataset.labels[self.indices[i]]
            self.classwise_indices[y].append(i)
        self.num_classes = max(self.classwise_indices.keys())+1

    def __getitem__(self, i):
        return self.base_dataset[self.indices[i]]

    def __len__(self):
        return len(self.indices)

    def get_class(self, i):
        return self.base_dataset.labels[self.indices[i]]

    # added for online update
    def set_dict(self, dict):
        self.classwise_indices = dict

    def get_dict(self):
        return self.classwise_indices


# variations of BatchSampler ===========================================================================================
class PairBatchSampler(Sampler): # for generating full batch
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.in_batch_size = batch_size // 2 # 64 for label batch
        self.in_in_batch_size = self.in_batch_size // 2 # 32 for source batch
        self.num_iterations = num_iterations
        self.source_len = self.dataset.base_dataset.source_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        source_len = self.dataset.base_dataset.source_len
        label_len = self.dataset.base_dataset.source_len + self.dataset.base_dataset.target_len
        target_len = self.dataset.base_dataset.target_len
        total_len = label_len + self.dataset.base_dataset.target_unl_len

        # for 1:1 = s:tl batch_indices
        source_indices = indices[:source_len]
        target_label_indices = indices[source_len:label_len]

        random.shuffle(source_indices)
        random.shuffle(target_label_indices)

        offset_s = 0
        offset_t = 0

        for k in range(len(self)):

            batch_indices_s = []
            batch_indices_t = []
            pair_indices = []

            while len(batch_indices_s) <= self.in_in_batch_size:
                if len(batch_indices_s) == self.in_in_batch_size:
                    break
                offset_s = offset_s % source_len
                source_index = source_indices[offset_s]
                y = self.dataset.get_class(source_index)
                filter_target = list(filter(lambda x: x >= label_len, self.dataset.classwise_indices[y]))

                if len(filter_target) == 0:
                    offset_s = offset_s + 1
                    continue
                else:
                    selected_pair = random.choice(filter_target)

                    batch_indices_s.append(source_index)
                    pair_indices.append(selected_pair)
                    offset_s = offset_s + 1

            while len(batch_indices_t) <= self.in_in_batch_size:
                if len(batch_indices_t) == self.in_in_batch_size:
                    break
                offset_t = offset_t % target_len
                target_index = target_label_indices[offset_t]
                y = self.dataset.get_class(target_index)
                filter_target = list(filter(lambda x: x >= label_len, self.dataset.classwise_indices[y]))

                if len(filter_target) == 0:
                    offset_t = offset_t + 1
                    continue
                else:
                    selected_pair = random.choice(filter_target)

                    batch_indices_t.append(target_index)
                    pair_indices.append(selected_pair)
                    offset_t = offset_t + 1

            batch_indices = batch_indices_s + batch_indices_t

            # for debugging
            assert (len(batch_indices_s) == self.in_in_batch_size and len(batch_indices_t) == self.in_in_batch_size)
            assert (min(batch_indices) >= 0) and (max(batch_indices) < label_len)
            assert (len(pair_indices) == self.in_batch_size)
            assert (min(pair_indices) >= label_len) and (max(pair_indices) < total_len)

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            # return (self.source_len + self.in_in_batch_size - 1) // self.in_in_batch_size # not drop last
            # for 1:1 source:target
            return self.source_len // self.in_in_batch_size # for drop last
        else:
            return self.num_iterations


class PairBatchSampler_for_uda(Sampler):
    def __init__(self, dataset, batch_size, num_iterations=None):
        self.dataset = dataset
        self.in_batch_size = batch_size // 2 # 64 for source batch
        self.num_iterations = num_iterations
        self.source_len = self.dataset.base_dataset.source_len

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        source_len = self.dataset.base_dataset.source_len
        unl_target_len = self.dataset.base_dataset.target_len
        total_len = source_len + unl_target_len

        # for 1:1 = s:tu batch_indices
        source_indices = indices[:source_len]

        random.shuffle(source_indices)

        offset_s = 0

        for k in range(len(self)):

            batch_indices_s = []
            pair_indices = []

            while len(batch_indices_s) <= self.in_batch_size:
                if len(batch_indices_s) == self.in_batch_size:
                    break
                offset_s = offset_s % source_len
                source_index = source_indices[offset_s]
                y = self.dataset.get_class(source_index)
                filter_target = list(filter(lambda x: x >= source_len, self.dataset.classwise_indices[y]))

                if len(filter_target) == 0:
                    offset_s = offset_s + 1
                    continue
                else:
                    selected_pair = random.choice(filter_target)

                    batch_indices_s.append(source_index)
                    pair_indices.append(selected_pair)
                    offset_s = offset_s + 1

            batch_indices = batch_indices_s

            # for debugging
            assert (len(batch_indices_s) == self.in_batch_size)
            assert (min(batch_indices) >= 0) and (max(batch_indices) < source_len)
            assert (len(pair_indices) == self.in_batch_size)
            assert (min(pair_indices) >= source_len) and (max(pair_indices) < total_len)

            yield batch_indices + pair_indices

    def __len__(self):
        if self.num_iterations is None:
            # return (self.source_len + self.in_in_batch_size - 1) // self.in_in_batch_size # not drop last
            # for 1:1 source:target
            return self.source_len // self.in_batch_size # for drop last
        else:
            return self.num_iterations


# make pseudo_label text file ==========================================================================================
def make_pseudo_label_rss(image_set_file_unl, pseudo_image_set_file_unl, G, F1, root, data_transforms, margin, args):

    G.eval()
    F1.eval()
    softmax = nn.Softmax(dim=1) # this is the original softmax

    batch_size = 512

    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['val'])
    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                                    batch_size=batch_size, num_workers=args.num_workers,
                                                    shuffle=False, drop_last=False)
    f = open(image_set_file_unl, 'r')
    pseudo_label_f = open(pseudo_image_set_file_unl, 'w')

    largest_margin = 0
    total_margin = 0

    margin_list = []

    # loader starts
    for i, (img, label) in enumerate(target_loader_unl):
        printProgress(i, len(target_loader_unl), 'Progress:', 'Complete', 1, 50)
        img = img.cuda()
        pseudo_label = torch.zeros(list(label.shape))

        with torch.no_grad():
            feature = G(img)
            output = F1(feature)
            pred = softmax(output)

        ### The margin between the first and the second most probability sample is applied
        top_two_class_index = torch.topk(output, 2)[1]
        top_two_class_prob = torch.topk(output, 2)[0]

        # find maximum prediction(softmax applied)
        max_pred = torch.max(pred, dim=1)[0]

        #assert(torch.equal(max_index, top_two_class_index[:,0]))
        # this could be false in extremely low probability

        # save pseudo label in txt file
        for j in range(len(pseudo_label)):
            line = f.readline()
            path = line.split()[0]

            cal_margin = top_two_class_prob[j,0] - top_two_class_prob[j,1]
            total_margin = total_margin + cal_margin

            margin_list.append(float(cal_margin)) # for histogram

            if cal_margin > largest_margin: # for logging
                largest_margin = cal_margin

            if cal_margin > margin:
                pseudo_label[j] = top_two_class_index[j,0]
            elif max_pred[j] > args.alpha_value:
                pseudo_label[j] = top_two_class_index[j,0]
            else:
                pseudo_label[j] = -1

            pseudo_label_f.write(path + ' ' + str(int(pseudo_label[j])) + '\n')

    mean_margin = total_margin * 1.0 / len(target_dataset_unl)

    f.close()
    pseudo_label_f.close()

    G.train()
    F1.train()

    return largest_margin, mean_margin, margin_list


# several utils ========================================================================================================
# calculate mean margin of all dataset
def calc_mean_margin(image_set_file_unl, G, F1, root, data_transforms, args):

    G.eval()
    F1.eval()

    batch_size = 800

    target_dataset_unl = Imagelists_VISDA(image_set_file_unl, root=root, transform=data_transforms['val'])
    target_loader_unl = torch.utils.data.DataLoader(target_dataset_unl,
                                                    batch_size=batch_size, num_workers=args.num_workers,
                                                    shuffle=False, drop_last=False)
    total_margin = 0

    for i, (img, label) in enumerate(target_loader_unl):
        printProgress(i, len(target_loader_unl), 'Progress:', 'Complete', 1, 50)
        img = img.cuda()

        with torch.no_grad():
            feature = G(img)
            output = F1(feature)

        top_two_class_prob = torch.topk(output, 2)[0]

        # pseudo_label = torch.max(output, 1)[1]
        # save pseudo label in txt file
        for j in range(len(label)):
            cal_margin = top_two_class_prob[j,0] - top_two_class_prob[j,1]
            total_margin = total_margin + cal_margin

    mean_margin = total_margin * 1.0 / len(target_dataset_unl)

    G.train()
    F1.train()

    return mean_margin
