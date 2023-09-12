import h5py
import numpy as np
from rdkit import Chem


# 作者将一些要用到的函数，统一写在了此文件中
def convert_to_smiles(vector, char):
    list_char = list(char)
    # list_char = char.tolist()
    vector = vector.astype(int)
    return "".join(map(lambda x: list_char[x], vector)).strip()  # 最后返回一个列表list_char用来存放生成的smiles,.strip()方法用来去除多余的空格


def stochastic_convert_to_smiles(vector, char):  # 传入一个向量后转化为字符串形式
    list_char = char.tolist()
    s = ""
    for i in range(len(vector)):
        prob = vector[i].tolist()
        norm0 = sum(prob)
        prob = [i / norm0 for i in prob]
        index = np.random.choice(len(list_char), 1, p=prob)
        s += list_char[index[0]]
    return s


def one_hot_array(i, n):  # 进行独热编码
    return list(map(int, [ix == i for ix in range(n)]))  # map函数可以传入多个可迭代对象


def one_hot_index(vec, charset):  # 获取每一字符的坐标
    return list(map(charset.index, vec))


def from_one_hot_array(vec):  # 以元组的形式返回vec中1的坐标
    oh = np.where(vec == 1)
    if oh[0].shape == (0,):
        return None
    return int(oh[0][0])


def decode_smiles_from_indexes(vec, charset):   #解码得到字符串（smiles）数据
    return "".join(map(lambda x: charset[x], vec)).strip()


def load_dataset(filename, split=True):
    h5f = h5py.File(filename, 'r')
    if split:
        data_train = h5f['data_train'][:]
    else:
        data_train = None
    data_test = h5f['data_test'][:]
    charset = h5f['charset'][:]
    h5f.close()
    if split:
        return data_train, data_test, charset
    else:
        return data_test, charset


def encode_smiles(smiles, model, charset):
    cropped = list(smiles.ljust(120))
    preprocessed = np.array([list(map(lambda x: one_hot_array(x, len(charset)), one_hot_index(cropped, charset)))])
    latent = model.encoder.predict(preprocessed)
    return latent


def smiles_to_onehot(smiles, charset):
    cropped = list(smiles.ljust(120))
    preprocessed = np.array([list(map(lambda x: one_hot_array(x, len(charset)), one_hot_index(cropped, charset)))])
    return preprocessed


def smiles_to_vector(smiles, vocab, max_length):
    while len(smiles) < max_length:
        smiles += " "
    return [vocab.index(str(x)) for x in smiles]


def decode_latent_molecule(latent, model, charset, latent_dim):
    decoded = model.decoder.predict(latent.reshape(1, latent_dim)).argmax(axis=2)[0]
    smiles = decode_smiles_from_indexes(decoded, charset)
    return smiles


def interpolate(source_smiles, dest_smiles, steps, charset, model, latent_dim):
    source_latent = encode_smiles(source_smiles, model, charset)
    dest_latent = encode_smiles(dest_smiles, model, charset)
    step = (dest_latent - source_latent) / float(steps)
    results = []
    for i in range(steps):
        item = source_latent + (step * i)
        decoded = decode_latent_molecule(item, model, charset, latent_dim)
        results.append(decoded)
    return results


def get_unique_mols(mol_list):
    inchi_keys = [Chem.InchiToInchiKey(Chem.MolToInchi(m)) for m in mol_list]
    u, indices = np.unique(inchi_keys, return_index=True)
    unique_mols = [[mol_list[i], inchi_keys[i]] for i in indices]
    return unique_mols


def accuracy(arr1, arr2, length):
    total = len(arr1)
    count1 = 0
    count2 = 0
    count3 = 0
    for i in range(len(arr1)):
        if np.array_equal(arr1[i, :length[i]], arr2[i, :length[i]]):
            count1 += 1  # 数组与数组进行比较，相似度越高，count值越高，准确率就越高
    for i in range(len(arr1)):
        for j in range(length[i]):
            if arr1[i][j] == arr2[i][j]:
                count2 += 1
            count3 += 1

    return float(count1 / float(total)), float(count2 / count3)


def extract_vocab(filename, seq_length):
    import collections
    with open(filename) as f:
        lines = f.read().split('\n')[:-1]
    lines = [l.split() for l in lines]
    lines = [l for l in lines if len(l[0]) < seq_length - 2]
    smiles = [l[0] for l in lines]

    total_string = ''
    for s in smiles:
        total_string += s
    counter = collections.Counter(total_string)
    count_pairs = sorted(counter.items(), key=lambda x: -x[1])
    chars, counts = zip(*count_pairs)
    vocab = dict(zip(chars, range(len(chars))))

    chars += ('E',)  # smiles片段的终止标识
    chars += ('X',)  # smiles片段的起始标识
    vocab['E'] = len(chars) - 2
    vocab['X'] = len(chars) - 1
    return chars, vocab


def load_data(filename, seq_length, char, vocab):
    with open(filename) as f:
        lines = f.read().split('\n')[:-1]
    smiles = [l for l in lines if len(l) < seq_length - 2]
    smiles_input = []
    smiles_output = []
    length = []
    for s in smiles:
        length.append(len(s) + 1)
        s1 = ('X' + s).ljust(seq_length, 'E')
        s2 = s.ljust(seq_length, 'E')
        list1 = list(map(vocab.get, s1))
        list2 = list(map(vocab.get, s2))
        if None in list1 or None in list2: continue
        smiles_input.append(list1)
        smiles_output.append(list2)
    smiles_input = np.array(smiles_input)
    smiles_output = np.array(smiles_output)
    length = np.array(length)

    return smiles_input, smiles_output, length
