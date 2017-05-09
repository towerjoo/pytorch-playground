import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

torch.manual_seed(1)

CONTEXT_SIZE = 2  # 2 words to the left, 2 to the right
raw_text = """We are about to study the idea of a computational process.
Computational processes are abstract beings that inhabit computers.
As they evolve, processes manipulate other abstract things called data.
The evolution of a process is directed by a pattern of rules
called a program. People create programs to direct processes. In effect,
we conjure the spirits of the computer with our spells.""".split()
word_to_ix = {}
for i, word in enumerate(raw_text):
    if word not in word_to_ix:
        word_to_ix[word] = len(word_to_ix)
data = []
# the original implementation will introduce some out-of-range index
# so let's make it consective to avoid such issue
for i in range(2, len(raw_text)-2):
    context = [raw_text[i-2], raw_text[i-1],
                raw_text[i+1], raw_text[i+2]]
    target = raw_text[i]
    data.append((context, target))

class CBOW(nn.Module):
    def __init__(self, vocab_size, embeding_dim, context_size):
        super(CBOW, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embeding_dim)
        self.linear1 = nn.Linear(embeding_dim * context_size, 128)
        self.linear2 = nn.Linear(128, vocab_size)

    def forward(self, inputs):
        #import ipdb;ipdb.set_trace()
        embeds = self.embedding(inputs).view(1, -1)
        out = F.relu(self.linear1(embeds))
        out = self.linear2(out)
        log_probs = F.log_softmax(out)
        return log_probs

losses = []
# here context_size should be 4 instead of 2
model = CBOW(len(word_to_ix), 10, 4)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=.001)

for epoch in range(10):
    total_loss = torch.Tensor([0])
    for context, target in data:
        model.zero_grad()
        context_vars = autograd.Variable(torch.LongTensor([word_to_ix[word] for word in context]))
        log_probs = model(context_vars)
        loss = loss_function(log_probs, autograd.Variable(torch.LongTensor([word_to_ix[target]])))
        loss.backward()
        optimizer.step()
        total_loss += loss.data
    losses.append(total_loss)
print losses
