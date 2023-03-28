import re
import string
from typing import Tuple
from copy import deepcopy
from torch.utils.data import Dataset
from .glove import glove6b_embedding

class SST(Dataset):
    def __init__(self, data: dict):
        super().__init__()
        self.data = data
        
    def __getitem__(self, index) -> tuple:
        t = self.data[index]
        return {
            "input": t[0],
            "label": t[1]
        }
    
    def __len__(self):
        return len(self.data)
    
    
class Tokenizer:
    def __init__(self, word2id: dict):
        self.word2id = word2id
        self.vocab_size = len(word2id)
        
    def tokenize_word(self, input):
        assert isinstance(input, str), f"Input {input} is not a string"
        if input in self.word2id:
            return self.word2id[input]
        else:
            # print(f"{input} is not in tokenizer dictionary")
            return self.vocab_size
        
        
    def tokenize_sentence(self, input: str):
        pattern = re.compile(f'[\s-]+')
        list_of_w = re.split(pattern, input.strip()) 
        result = list()
        for w in list_of_w:
            result.append(self.tokenize_word(w.lower()))
        return result
        
        
    def tokenize(self, input):
        """
            Support word, list of word, list of list of word
        """
        if isinstance(input, str):
            return self.tokenize_word(input)
        elif isinstance(input, list):
            result = list()
            for word in input:
                if isinstance(word, str):
                    result.append(self.tokenize_word(word))
                elif isinstance(word, list):
                    tmp = list()
                    for w in word:
                        tmp.append(self.tokenize_word(w))
                    result.append(tmp)
        else:
            raise TypeError(f"Unexpected input type {type(input)}")
        return result


def process_line_data(line) -> Tuple:
    pattern = re.compile(r'\(([^\(\)]+)\)')
    score = line[1]
    matches = re.findall(pattern, line)
    while any('(' in match or ')' in match for match in matches):
        matches = [re.findall(pattern, match)[0] if '(' in match else match for match in matches]
    return " ".join([match[2:] for match in matches]), int(score)


def prepare_datasets(name):
    data = open(f'data/trees/{name}.txt').read().strip().split('\n')
    sentence = list()
    pattern = re.compile(r'\(([^\(\)]+)\)')
    for line in data:
        sentence.append(
            process_line_data(line)
        )    
    return SST(sentence)


def prepare_tokenizer_embedding(glove_path):
    embedding, id2word = glove6b_embedding(glove_path, dim=100)
    word2id = dict()
    for idx, word in enumerate(id2word):
        word2id[word] = idx
    tokenizer = Tokenizer(word2id)
    return tokenizer, embedding, id2word, word2id


def main():
    pass
    
    
if __name__ == '__main__':
    main()