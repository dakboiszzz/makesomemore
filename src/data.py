import torch

# Create DataPrep class, set block_size
class DataPrep():
    def __init__(self,words,block_size = 2):
        self.words = words
        self.block_size = block_size
    def getChars(self):
        self.chars = []
        ## This is my code but it's not really good
        """
        for word in self.words:
            for char in list(word):
                if char not in self.chars:
                    self.chars.append(char)
        self.chars = sorted(self.chars)
        """
        self.chars = sorted(list(set(''.join(self.words))))
        return self.chars
    # Create a mapping from string to int, also a reverse mapping to turn int back to string
    def string_to_int(self):
        stoi = dict()
        chars = self.getChars()
        for index,char in enumerate(chars):
            stoi[char] = index + 1
        stoi['.'] = 0
        self.stoi = stoi
        return self.stoi
    def int_to_string(self):
        stoi = self.string_to_int()
        self.itos = {i:s for s,i in stoi.items()}
        return self.itos
    def getData(self):
        chars = self.getChars()

        stoi = self.string_to_int()
        itos = self.int_to_string()
        X = []
        Y = []
        for word in self.words:
            block = [0] * self.block_size
            word = word + '.'
            for char in word:
                X.append(block)
                index = stoi[char]
                Y.append(index)
                block = block[1:] + [index]
        X = torch.tensor(X)
        Y = torch.tensor(Y)
        return X,Y
    # Load the data set
    @staticmethod
    def load_words(filepath):
        """Load words from file"""
        with open(filepath, 'r') as f:
            return f.read().splitlines()