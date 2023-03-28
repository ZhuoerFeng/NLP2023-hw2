import numpy as np

def glove6b_from_txt(fname) -> dict:
    f = open(fname).read().strip().split('\n')
    word2vec_map = dict()
    for line in f:
        line = line.strip().split(' ')
        word = line[0]
        vec = np.asarray(line[1:], dtype=np.float32)
        word2vec_map[word] = vec
    return word2vec_map

def glove6b_embedding(fname, dim=50):
    id2word = list()
    word2vec_map = glove6b_from_txt(fname)
    embedding = np.zeros((len(word2vec_map) + 1, dim), dtype=np.float32)
    cnt = 0
    for k, v in word2vec_map.items():
        id2word.append(k)
        embedding[cnt] = v
        cnt += 1
    embedding[len(word2vec_map)] = np.random.uniform(-1, 1, size=(dim))
    return embedding, id2word
    
    
if __name__ == '__main__':
    glove6b_from_txt('glove/glove.6B.50d.txt')