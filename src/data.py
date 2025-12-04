import torch

# Create DataPrep class, set block_size
class DataPrep():
    def __init__(self,words,block_size = 2):
        self.words = words
        self.block_size = block_size
        self.stoi = self.string_to_int()
        self.itos = self.int_to_string()
        self.chars = self.getChars()
        self.vocab_size = len(self.chars)
    def getChars(self):
        ## This is my code but it's not really good
        """
        for word in self.words:
            for char in list(word):
                if char not in self.chars:
                    self.chars.append(char)
        self.chars = sorted(self.chars)
        """
        return sorted(list(set(''.join(self.words))))
    # Create a mapping from string to int, also a reverse mapping to turn int back to string
    def string_to_int(self):
        stoi = dict()
        for index,char in enumerate(self.chars):
            stoi[char] = index + 1
        stoi['.'] = 0
        return stoi
    def int_to_string(self):
        stoi = self.stoi
        itos = {i:s for s,i in stoi.items()}
        return itos
    def getData(self):
        X = []
        Y = []
        for word in self.words:
            block = [0] * self.block_size
            word = word + '.'
            for char in word:
                X.append(block)
                index = self.stoi[char]
                Y.append(index)
                block = block[1:] + [index]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X,Y