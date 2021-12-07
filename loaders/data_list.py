import numpy as np
import os
import os.path
from PIL import Image


def pil_loader(path):
    with open(path, 'rb') as f:
        img = Image.open(f)
        return img.convert('RGB')


def make_dataset_fromlist(image_list):
    with open(image_list) as f:
        image_index = [x.split(' ')[0] for x in f.readlines()]
    with open(image_list) as f:
        label_list = []
        selected_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[1].strip()
            label_list.append(int(label))
            selected_list.append(ind)
        image_index = np.array(image_index)
        label_list = np.array(label_list)
    image_index = image_index[selected_list]
    return image_index, label_list


def return_classlist(image_list):
    data_num = 0
    with open(image_list) as f:
        label_list = []
        for ind, x in enumerate(f.readlines()):
            label = x.split(' ')[0].split('/')[-2]
            if label not in label_list:
                label_list.append(str(label))
            data_num = data_num + 1

    return label_list, data_num


class Imagelists_VISDA(object):
    def __init__(self, image_list, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        imgs, labels = make_dataset_fromlist(image_list)
        self.imgs = imgs
        self.labels = labels
        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


# concat source, labeled target, and unlabeled target dataset
class Imagelists_concat3(object):
    def __init__(self, image_list1, image_list2, image_list3, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        imgs1, labels1 = make_dataset_fromlist(image_list1) # for source
        imgs2, labels2 = make_dataset_fromlist(image_list2) # for labeled target
        imgs3, labels3 = make_dataset_fromlist(image_list3) # for unlabeled target

        imgs = np.concatenate((imgs1, imgs2, imgs3), axis=None)
        labels = np.concatenate((labels1, labels2, labels3), axis=None)

        self.imgs = imgs
        self.labels = labels

        self.source_len = len(imgs1)
        self.target_len = len(imgs2)
        self.target_unl_len = len(imgs3)

        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test: # if train
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)


# concat two types of dataset
class Imagelists_concat2(object):
    def __init__(self, image_list1, image_list2, root="./data/multi/",
                 transform=None, target_transform=None, test=False):
        imgs1, labels1 = make_dataset_fromlist(image_list1) # for source
        imgs2, labels2 = make_dataset_fromlist(image_list2) # for labeled target

        imgs = np.concatenate((imgs1, imgs2), axis=None)
        labels = np.concatenate((labels1, labels2), axis=None)

        self.imgs = imgs
        self.labels = labels

        self.source_len = len(imgs1)
        self.target_len = len(imgs2) # this will be unl target len in uda exp

        self.transform = transform
        self.target_transform = target_transform
        self.loader = pil_loader
        self.root = root
        self.test = test

    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is
            class_index of the target class.
        """
        path = os.path.join(self.root, self.imgs[index])
        target = self.labels[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)
        if not self.test:
            return img, target
        else:
            return img, target, self.imgs[index]

    def __len__(self):
        return len(self.imgs)

