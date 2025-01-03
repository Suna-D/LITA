from torch.utils.data import Dataset
import torchvision.transforms as transforms
import cv2 as cv
import os
from tqdm import tqdm
import pandas as pd

mean = [0.485, 0.456, 0.406]  # RGB
std = [0.229, 0.224, 0.225]
    
class OriginalBBDataset(Dataset):
    def __init__(self, file_dir='dataset', type='train', test=False):
        self.type = type
        self.if_test = test
        self.train_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.test_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.images = []
        self.pic_paths = []
        self.labels = []

        if type == 'train':
            DATA = pd.read_csv(os.path.join(file_dir, 'train_set.csv'))

        labels = DATA['score'].values.tolist()
        pic_paths = DATA['image'].values.tolist()
        for i in tqdm(range(len(pic_paths))):
            pic_path = os.path.join('BAID/images', pic_paths[i])
            label = float(labels[i])
            self.pic_paths.append(pic_path)
            self.labels.append(label)
        
    def __len__(self):
        return len(self.pic_paths)
    
    def __getitem__(self, index):
        pic_path = self.pic_paths[index]
        img = cv.imread(pic_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.train_transformer(img)
    
        return pic_path, img, self.labels[index]

class BBDataset(Dataset):
    def __init__(self, file_dir='dataset', type='train', test=False):
        self.type = type
        self.if_test = test
        self.train_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.test_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.images = []
        self.pic_paths = []
        self.labels = []

        if type == 'train':
            DATA = pd.read_csv(os.path.join(file_dir, 'box_cox_train_set.csv'))

        labels = DATA['score'].values.tolist()
        pic_paths = DATA['image'].values.tolist()
        for i in tqdm(range(len(pic_paths))):
            pic_path = os.path.join('BAID/images', pic_paths[i])
            label = float(labels[i])
            self.pic_paths.append(pic_path)
            self.labels.append(label)
        self.aesthetics_llava = pd.read_csv('aesthetics_comments.csv', header=None)
        self.style_llava =  pd.read_csv('style_comments.csv', header=None)

    def __len__(self):
        return len(self.pic_paths)
    
    def __getitem__(self, index):
        pic_path = self.pic_paths[index]
        img = cv.imread(pic_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.train_transformer(img)
    
        return pic_path, img, self.labels[index], self.aesthetics_llava.iloc[index, 0], self.style_llava.iloc[index, 0]

class BBTestDataset(Dataset):
    def __init__(self, file_dir, type, test=False):
        self.type = type
        self.if_test = test
        self.train_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.test_transformer = transforms.Compose(
            [
                transforms.ToPILImage(),
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=mean, std=std),
            ]
        )

        self.images = []
        self.pic_paths = []
        self.labels = []

        if type == 'validation':
            DATA = pd.read_csv(os.path.join(file_dir, 'val_set.csv'))
        elif type == 'test':
            DATA = pd.read_csv(os.path.join(file_dir, 'test_set.csv'))

        labels = DATA['score'].values.tolist()
        pic_paths = DATA['image'].values.tolist()
        for i in tqdm(range(len(pic_paths))):
            pic_path = os.path.join('BAID/images', pic_paths[i])
            label = float(labels[i])
            self.pic_paths.append(pic_path)
            self.labels.append(label)

    def __len__(self):
        return len(self.pic_paths)

    def __getitem__(self, index):
        pic_path = self.pic_paths[index]
        img = cv.imread(pic_path)
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = self.test_transformer(img)

        return pic_path, img, self.labels[index]
