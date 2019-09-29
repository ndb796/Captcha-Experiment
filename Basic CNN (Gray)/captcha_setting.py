import os
from torch.utils.data import DataLoader,Dataset
import torchvision.transforms as transforms
from PIL import Image
import numpy as np


# character_set = '0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ'
character_set = '0123456789'
character_set_length = len(character_set)

number_per_image = 4
width = 40 + 20 * number_per_image
height = 100

train_dataset_path = 'dataset' + os.path.sep + 'train'
test_dataset_path = 'dataset' + os.path.sep + 'test'

transform = transforms.Compose([
    transforms.Grayscale(),
    transforms.ToTensor(),
])


class Dataset():
    def __init__(self, folder, transform=None):
        self.train_image_file_paths = [os.path.join(folder, image_file) for image_file in os.listdir(folder)]
        self.transform = transform

    def __len__(self):
        return len(self.train_image_file_paths)

    def __getitem__(self, idx):
        image_root = self.train_image_file_paths[idx]
        image_name = image_root.split(os.path.sep)[-1]
        image = Image.open(image_root)
        if self.transform is not None:
            image = self.transform(image)
        label = encode(image_name.split('_')[0])
        return image, label


def get_train_data_loader():
    dataset = Dataset(train_dataset_path, transform=transform)
    return DataLoader(dataset, batch_size=64, shuffle=True)


def get_test_data_loader():
    dataset = Dataset(test_dataset_path, transform=transform)
    return DataLoader(dataset, batch_size=1, shuffle=True)


def char2pos(c):
    if c == '_':
        k = 62
        return k
    k = ord(c) - 48
    if k > 9:
        k = ord(c) - 65 + 10
        if k > 35:
            k = ord(c) - 97 + 26 + 10
            if k > 61:
                raise ValueError('error')
    return k


def encode(text):
    vector = np.zeros(character_set_length * number_per_image, dtype=float)
    for i, c in enumerate(text):
        idx = i * character_set_length + char2pos(c)
        vector[idx] = 1.0
    return vector


def decode(vec):
    char_pos = vec.nonzero()[0]
    text = []
    for i, c in enumerate(char_pos):
        char_idx = c % character_set_length
        if char_idx < 10:
            char_code = char_idx + ord('0')
        elif char_idx < 36:
            char_code = char_idx - 10 + ord('A')
        elif char_idx < 62:
            char_code = char_idx - 36 + ord('a')
        elif char_idx == 62:
            char_code = ord('_')
        else:
            raise ValueError('error')
        text.append(chr(char_code))
    return "".join(text)


# e = encode("8937")
# print(e)
# print(decode(e))
