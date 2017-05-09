import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

data = [("me gusta comer en la cafeteria".split(), "SPANISH"),
    ("Give it to me".split(), "ENGLISH"),
    ("No creo que sea una buena idea".split(), "SPANISH"),
    ("No it is not a good idea to get lost at sea".split(), "ENGLISH")]

test_data = [("Yo creo que si".split(), "SPANISH"),
    ("it is lost on me".split(), "ENGLISH")]

word_to_ix = {}
for sent, _ in data + test_data:
    for word in sent:
        if word not in word_to_ix:
            word_to_ix[word] = len(word_to_ix)
print (word_to_ix)
VOCAB_SIZE = len(word_to_ix)
NUM_LABELS = 2


class BOWClassifier(nn.Module):
    def __init__(self, num_labels, vocab_size):
        super(BOWClassifier, self).__init__()
        self.linear = nn.Linear(vocab_size, num_labels)

    def forward(self, bow_vec):
        return F.log_softmax(self.linear(bow_vec))

def make_bow_vector(sentence, word_to_ix):
    vec = torch.zeros(len(word_to_ix))
    for word in sentence:
        vec[word_to_ix[word]] += 1
    return vec.view(1, -1)

def make_target(label, label_to_ix):
    return torch.LongTensor([label_to_ix[label]])

model = BOWClassifier(NUM_LABELS, VOCAB_SIZE)
label_to_ix = {"SPANISH": 0, "ENGLISH": 1}

def print_test_probs():
    for instance, label in test_data:
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        log_probs = model(bow_vec)
        print(log_probs)

def get_prob(model):
    params = next(model.parameters())
    words = ["creo", "me", "gusta", "lost", "at"]
    for word in words:
        print("{} with prob: {}".format(word, params[:, word_to_ix[word]]))

get_prob(model)

loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=.1)
for epoch in range(100):
    for instance, label in data:
        model.zero_grad()
        bow_vec = autograd.Variable(make_bow_vector(instance, word_to_ix))
        target = autograd.Variable(make_target(label, label_to_ix))
        log_probs = model(bow_vec)
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()


get_prob(model)
