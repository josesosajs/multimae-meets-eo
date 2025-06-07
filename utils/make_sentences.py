import numpy as np
import random
import os
import json
from transformers import DistilBertModel, DistilBertConfig, DistilBertTokenizer
import h5py


def get_class(class_def_path):
    with open(class_def_path) as json_file: data = json.load(json_file)

    return dict((v, k) for k, v in data.items())

def format_sentence(class_esa):
    classes_names_esa = ""
        
    if len(class_esa) == 1: classes_names_esa = class_esa[0]
    elif len(class_esa) == 2: classes_names_esa = '{} and {}'.format(class_esa[0], class_esa[1])
    else:
        for i, item in enumerate(class_esa):
            if i == 0: classes_names_esa = classes_names_esa + '{},'.format(item)
            elif i == len(class_esa)-1: classes_names_esa = classes_names_esa + 'and {}'.format(item)
            else: classes_names_esa = classes_names_esa + '{},'.format(item)

    return classes_names_esa

def decode_month(hdf5_file, i):
    month_sin = (6 * np.arcsin(abs(hdf5_file['month'][i][0]))) / np.pi
    month_cos = (6 * np.arccos(abs(hdf5_file['month'][i][1]))) / np.pi

    return month_sin, month_cos

def get_encoded_sentence(sentence, tokenizer):
    max_length = 60
    encoded_caption = tokenizer(list(sentence), padding="max_length", truncation=True, max_length=max_length)

    return encoded_caption


def get_sentence(data, i, tokenizer):
    eco_labels = dict(get_class('/project/home/p200249/jsosa/mmearth_data/data_1M_v001/eco_labels.json'))
    biome_labels = dict(get_class('/project/home/p200249/jsosa/mmearth_data/data_1M_v001/biome_labels.json'))

    class_names_esa = {0: 'No data', 10: 'Tree cover', 20:'Shrubland', 30: 'Grassland',
                       40: 'Cropland', 50: 'Built-up', 60: 'Sparse vegetation', 70: 'Snow and ice',
                       80: 'Permanent water bodies', 90: 'Herbaceous wetland', 95: 'Mangroves',
                       100: 'Moss and lichen'}

    freq_dict = {}
    esa_wordlcover = data['esa_worldcover'][i]
    freq = np.array(np.unique(esa_wordlcover, return_counts=True)).T
        
    for item in freq:
        if int(item[1])*100/(64*64) >=20.:
            freq_dict[int(item[0])] = int(item[1])*100/(64*64)

    class_esa = [class_names_esa[key] for key, value in freq_dict.items()]
    classes_names_esa = format_sentence(class_esa)
        
    biome = data['biome'][i]
    biome_name = biome_labels[np.where(biome==1)[0][0]].replace('&', 'and')

    eco_region = data['eco_region'][i]
    eco_region_name = eco_labels[np.where(eco_region==1)[0][0]].replace('&', 'and')

    sentence = "A satellite photo of the {}, showing {} in a {} biome".format(eco_region_name,classes_names_esa, biome_name).lower()
    encoded_sentence = get_encoded_sentence([sentence], tokenizer)

    return sentence, encoded_sentence

# if __name__ == "__main__":

#     tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
#     data_path = '/project/home/p200249/jsosa/mmearth_data/data_1M_v001_64/data_1M_v001_64.h5'
#     data = h5py.File(data_path, 'r')
#     #index = 98934
#     print(len(data['metadata']))

#     for index in range(len(data['metadata'])):
#         sentence, encoded_sentence = get_sentence(data, index, tokenizer)
#         print(sentence, encoded_sentence)







