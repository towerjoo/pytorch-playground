import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

def prepare_sequence(seq, to_ix):
    idxs = [to_ix[w] for w in seq]
    tensor = torch.LongTensor(idxs)
    return autograd.Variable(tensor)

training_data = [
    ("The dog ate the apple".split(), ["DET", "NN", "V", "DET", "NN"]),
    ("Everybody read that book".split(), ["NN", "V", "DET", "NN"])
]

word_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)

char_to_ix = {}
for sent, tags in training_data:
    for word in sent:
        for char in word:
            if char not in char_to_ix:
                char_to_ix[char] = len(char_to_ix)

tag_to_ix = {"DET": 0, "NN": 1, "V": 2}

EMBEDDING_DIM = 6
HIDDEN_DIM = 6

class CharLSTMTagger(nn.Module):
    def __init__(self, embedding_dim, hidden_dim, vocab_size, tagset_size, char_size):
        super(CharLSTMTagger, self).__init__()

        self.hidden_dim = hidden_dim
        self.word_embeddings = nn.Embedding(vocab_size, embedding_dim)
        # for char lstm we also use hidden_dim, we can also use
        # a different hyperparam here
        self.lstm = nn.LSTM(embedding_dim+hidden_dim, hidden_dim)
        self.hidden2tag = nn.Linear(hidden_dim, tagset_size)
        self.hidden = self.init_hidden()

        self.char_embeddings = nn.Embedding(char_size, embedding_dim)
        self.char_lstm = nn.LSTM(embedding_dim, hidden_dim)


    def init_hidden(self):
        return (autograd.Variable(torch.zeros(1, 1, self.hidden_dim)),
                autograd.Variable(torch.zeros(1, 1, self.hidden_dim)))

    def get_char_repr(self, word):
        hidden = self.init_hidden()
        word_var = prepare_sequence(word, char_to_ix)
        embeds = self.char_embeddings(word_var)
        _, h = self.char_lstm(embeds.view(len(word), 1, -1), hidden)
        return h[0].view(1, -1)

    def forward(self, sentence, orig_sent):
        embeds = self.word_embeddings(sentence)
        for word in orig_sent:
            char_repr = self.get_char_repr(word)
            embeds = torch.cat((embeds, char_repr))
        inpt = embeds.view(len(sentence), 1, -1)
        lstm_out, self.hidden = self.lstm(inpt, self.hidden)
        tag_space = self.hidden2tag(lstm_out.view(len(sentence), -1))
        tag_scores = F.log_softmax(tag_space)
        return tag_scores


model = CharLSTMTagger(EMBEDDING_DIM, HIDDEN_DIM, len(word_to_ix), len(tag_to_ix), len(char_to_ix))
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=.1)

inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs, training_data[0][0])
print(tag_scores)

for epoch in range(300):
    for sentence, tags in training_data:
        model.zero_grad()
        model.hidden = model.init_hidden()
        sentence_in = prepare_sequence(sentence, word_to_ix)
        targets = prepare_sequence(tags, tag_to_ix)

        tag_scores = model(sentence_in, sentence)
        loss = loss_function(tag_scores, targets)
        loss.backward()
        optimizer.step()

inputs = prepare_sequence(training_data[0][0], word_to_ix)
tag_scores = model(inputs, training_data[0][0])
print(tag_scores)

inputs = prepare_sequence(training_data[1][0], word_to_ix)
tag_scores = model(inputs, training_data[1][0])
print(tag_scores)
