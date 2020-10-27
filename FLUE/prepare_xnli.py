import os

script_dir = os.path.dirname(os.path.realpath(__file__))
path_bash_script = os.path.join(script_dir, 'get-data-xnli.sh')

os.system('bash {}'.format(path_bash_script))

path = os.path.join('processed')
path_new = 'XNLI'


datasets = [('fr.raw.valid', 'valid'), ('fr.raw.test', 'test'), ('fr.raw.train', 'train')]

def write_data(path, data):
    with open(path, 'w+') as fw:
        for element in data:
            fw.write('{}\n'.format(element))
        fw.close()

labels = ['contradiction', 'entailment', 'neutral']
for dataset in datasets:
    with open(os.path.join(path, dataset[0]), 'r') as fr:
        fr.readline() #skip first line
        examples = []
        for line in fr:
            example = line.split('\t')
            example[-1] = example[-1].strip()
            assert example[-1] in labels
            example[-1] = labels.index(example[-1])
            examples.append(example)
        write_data(os.path.join(path_new, '{}.sent1'.format(dataset[1])),
                  [example[0] for example in examples])
        write_data(os.path.join(path_new, '{}.sent2'.format(dataset[1])),
                  [example[1] for example in examples])
        write_data(os.path.join(path_new, '{}.label'.format(dataset[1])),
                   [example[2] for example in examples])
        fr.close()
        
os.system('rm -rf processed raw')
