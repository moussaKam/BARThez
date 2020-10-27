import os

os.system('wget https://storage.googleapis.com/paws/pawsx/x-final.tar.gz')

os.system('tar -xf x-final.tar.gz; rm x-final.tar.gz')


path = os.path.join('x-final', 'fr')
path_new = 'PAWSX'

os.system('mkdir {}'.format(path_new))

datasets = [('dev_2k.tsv', 'valid'), ('test_2k.tsv', 'test'), ('translated_train.tsv', 'train')]

def write_data(path, data):
    with open(path, 'w+') as fw:
        for element in data:
            fw.write('{}\n'.format(element))
        fw.close()

for dataset in datasets:
    with open(os.path.join(path, dataset[0]), 'r') as fr:
        fr.readline() #skip first line
        examples = []
        for line in fr:
            examples.append(line.split('\t'))
        write_data(os.path.join(path_new, '{}.sent1'.format(dataset[1])),
                  [example[1] for example in examples])
        write_data(os.path.join(path_new, '{}.sent2'.format(dataset[1])),
                  [example[2] for example in examples])
        write_data(os.path.join(path_new, '{}.label'.format(dataset[1])),
                  [example[3][0] for example in examples])
    fr.close()
    
os.system('rm -rf x-final')
