import numpy as np
import os
import pickle
from PIL import Image

data_dir = 'by_class'
home_dir = os.getcwd()
hex_chars = ['30', '31', '32', '33', '34', '35', '36', '37', '38', '39']

def load_pickle_file(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

def save_pickle_file(filename, data):
    with open(filename, 'wb') as f:
        pickle.dump(data, f)

def hex_to_decimal(hex_char):
    return int(hex_char, 16)

def get_images_from_dir(images_dir):
    filenames = os.listdir(images_dir)
    images = np.empty([len(filenames), 128, 128], dtype=np.uint8)

    for i, f in enumerate(filenames):
        image_dir = f'{images_dir}/{f}'
        image = Image.open(image_dir).convert('L')
        image = np.array(image)
        
        images[i] = image

    return images

def get_num_images_in_dir(directory):
    filenames = os.listdir(directory)
    return len(filenames)

def get_num_images(typ):
    num_train_images = 0
    for hex_char in hex_chars:
        if typ == 'train':
            directory = f'{home_dir}/{data_dir}/{hex_char}/train_{hex_char}'
        elif typ == 'test':
            directory = f'{home_dir}/{data_dir}/{hex_char}/hsf_4'
        else:
            return
        num_train_images += get_num_images_in_dir(directory)
    
    return num_train_images

def save_images(typ):
    if typ not in ['train', 'test']: return

    num_total_images = get_num_images(typ)

    filename = f'nist_data/{typ}_images.pkl'

    all_images = np.empty([num_total_images, 128, 128], dtype=np.uint8)

    save_pickle_file(filename, all_images)

    total_count = 0
    for hex_char in hex_chars:
        all_images = load_pickle_file(filename)

        if typ == 'train':
            directory = f'{home_dir}/{data_dir}/{hex_char}/train_{hex_char}'
        elif typ == 'test':
            directory = f'{home_dir}/{data_dir}/{hex_char}/hsf_4'

        images = get_images_from_dir(directory)
        num_images_for_hex = np.shape(images)[0]
        print(num_images_for_hex)

        all_images[total_count:(total_count+num_images_for_hex)] = images

        save_pickle_file(filename, all_images)

        total_count += num_images_for_hex

        print('finished', hex_char)

save_images('train')
