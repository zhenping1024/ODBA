import os
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from collections import Counter

class GTSRBDataset(Dataset):
    def __init__(self, root_dir, csv_file=None, transform=None):
        self.root_dir = root_dir
        if csv_file:
            self.annotations = self._load_annotations(csv_file)
        else:
            self.annotations = self._load_annotations_from_dir(root_dir)
        self.transform = transform

    def _load_annotations(self, csv_file):
        df = pd.read_csv(csv_file, delimiter=';')
        return df[['Filename', 'ClassId']]

    def _load_annotations_from_dir(self, root_dir):
        all_data = []
        for class_dir in os.listdir(root_dir):
            class_path = os.path.join(root_dir, class_dir)
            if os.path.isdir(class_path):
                for img_file in os.listdir(class_path):
                    if img_file.endswith(('.ppm', '.png', '.jpg', '.jpeg')):
                        all_data.append([os.path.join(class_dir, img_file), class_dir])
        return pd.DataFrame(all_data, columns=['Filename', 'ClassId'])

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, index):
        img_file = self.annotations.iloc[index, 0]
        label = int(self.annotations.iloc[index, 1])
        img_path = os.path.join(self.root_dir, img_file)
        image = Image.open(img_path)
        
        if self.transform:
            image = self.transform(image)

        return image, label


