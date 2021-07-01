import numpy as np
import gensim

def build_word2id(train_path, val_path, test_path):
    word2id = {'PAD': 0}
    paths = [train_path, val_path, test_path]
    for path in paths:
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                sp = line.strip().split()
                for word in sp[1:]:
                    if word not in word2id.keys():
                        word2id[word] = len(word2id)

    tag2id = {'0': 0, '1': 1}

    return word2id, tag2id

def build_word2vec(word2id, args):
    n_words = max(word2id.values()) + 1
    model = gensim.models.KeyedVectors.load_word2vec_format(args.w2v_data_path, binary=True)
    word_vecs = np.array(np.random.uniform(-1., 1., [n_words, model.vector_size]))
    for word in word2id.keys():
        try:
            word_vecs[word2id[word]] = model[word]
        except KeyError:
            pass
    if args.save_w2v:
        with open(args.save_w2v, 'w', encoding='utf-8') as f:
            for vec in word_vecs:
                vec = [str(v) for v in vec]
                f.write(' '.join(vec))
                f.write('\n')

    return word_vecs

def load_word2vec(save_w2v_path):
    word_vecs_list = []
    with open(save_w2v_path, 'r', encoding='utf-8') as f:
        word_vecs = f.read()
        word_vecs = word_vecs.split('\n')[0:-1]
        for vec in word_vecs:
            vec = vec.split(' ')
            vec = [float(v) for v in vec]
            word_vecs_list.append(vec)
        word_vecs = np.array(word_vecs_list)
    
    return word_vecs


def get_train_data(train_path, val_path, word2id, tag2id, max_length=60):
    x_train_id, x_val_id = [], []
    y_train_id, y_val_id = [], []

    with open(train_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            data = line.strip().split()
            y_train_id.append(data[0])
            content = [word2id.get(d,0) for d in data[1:]]
            content = content[:max_length]
            if len(content) < max_length:
                content += [word2id['PAD']] * (max_length - len(content))
            x_train_id.append(content)

    x_train = np.asarray(x_train_id).astype(np.int64)
    y_train = np.asarray(y_train_id).astype(np.int64)

    with open(val_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            data = line.strip().split()
            y_val_id.append(data[0])
            content = [word2id.get(d,0) for d in data[1:]]
            content = content[:max_length]
            if len(content) < max_length:
                content += [word2id['PAD']] * (max_length - len(content))
            x_val_id.append(content)

    x_val = np.asarray(x_val_id).astype(np.int64)
    y_val = np.asarray(y_val_id).astype(np.int64)

    return x_train, y_train, x_val, y_val

def get_test_data(test_path, word2id, tag2id, max_length=60):
    x_test_id, y_test_id = [], []

    with open(test_path, 'r', encoding='UTF-8') as f:
        for line in f.readlines():
            data = line.strip().split()
            y_test_id.append(data[0])
            content = [word2id.get(d,0) for d in data[1:]]
            content = content[:max_length]
            if len(content) < max_length:
                content += [word2id['PAD']] * (max_length - len(content))
            x_test_id.append(content)

    x_test = np.asarray(x_test_id).astype(np.int64)
    y_test = np.asarray(y_test_id).astype(np.int64)

    return x_test, y_test