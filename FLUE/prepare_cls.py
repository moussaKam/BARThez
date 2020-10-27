import os
import xml.etree.ElementTree as ET
import random

cls_url = 'https://zenodo.org/record/3251672/files/cls-acl10-unprocessed.tar.gz'

os.system('wget {}'.format(cls_url))
os.system('tar -xf cls-acl10-unprocessed.tar.gz; rm cls-acl10-unprocessed.tar.gz')

assert os.path.isdir('cls-acl10-unprocessed')

valid_size = 0.1

categories = ['books', 'dvd', 'music']
path = 'cls-acl10-unprocessed/fr'

path_new = '.'
#os.system('mkdir {}'.format(path_new))

def write_data(path, data):
    with open(path, 'w+') as f:
        for element in data:
            f.write('{}\n'.format(element))
        f.close()

for category in categories:
    path_cat = os.path.join(path, category)
    os.system('mkdir {}'.format(os.path.join(path_new, category)))
    
    for dataset in ['train.review', 'test.review']:
        tree = ET.parse(os.path.join(path_cat, dataset))
        root = tree.getroot()
        reviews = []
        for item in root.findall('item'):
            review = item.find('text').text
            if category=='dvd':
                review+=' {}'.format(item.find('title').text)
            review = review.replace('\n', ' ')
            rating = int(item.find('rating').text[0])
            label = 1 if rating > 3 else 0
            reviews.append((review, label))
            
        if dataset == 'train.review':
            pos_reviews = [review for review in reviews if review[1]==1]
            neg_reviews = [review for review in reviews if review[1]==0]
            
            random.shuffle(pos_reviews)
            random.shuffle(neg_reviews)
            
            n_valid_pos = int(valid_size*len(pos_reviews))
            n_valid_neg = int(valid_size*len(neg_reviews))
            
            valid_reviews = pos_reviews[:n_valid_pos] + neg_reviews[:n_valid_neg]
            train_reviews = pos_reviews[n_valid_pos:] + neg_reviews[n_valid_neg:]
            
            random.shuffle(pos_reviews)
            random.shuffle(neg_reviews)
            
            write_data(os.path.join(path_new, category, 'valid.review'),
                       [el[0] for el in valid_reviews])
            write_data(os.path.join(path_new, category, 'valid.label'),
                       [el[1] for el in valid_reviews])
            write_data(os.path.join(path_new, category, 'train.review'),
                       [el[0] for el in train_reviews])
            write_data(os.path.join(path_new, category, 'train.label'),
                       [el[1] for el in train_reviews])
        
        else:
            write_data(os.path.join(path_new, category, 'test.review'),
                       [el[0] for el in reviews])
            write_data(os.path.join(path_new, category, 'test.label'),
                       [el[1] for el in reviews])

os.system('rm -rf cls-acl10-unprocessed')
