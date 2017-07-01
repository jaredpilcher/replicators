import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

NUM_INPUTS = 2
NUM_OUTPUTS = 2
NUM_EPOCHS = 100
LEARNING_RATE = 0.1

data = [([0,0], [0]),
        ([0,1], [1]),
        ([1,0], [1]),
        ([1,1], [0])]

test_data = [([0,0], [0]),
             ([0,1], [1])]


class Classifier(nn.Module):

    def __init__(self, num_outputs, num_inputs):
        super(Classifier, self).__init__()
        self.linear = nn.Linear(num_inputs, num_outputs)


    def forward(self, input_vector):
        return F.log_softmax(self.linear(input_vector))



model = Classifier(NUM_OUTPUTS, NUM_INPUTS)
loss_function = nn.NLLLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE)

def replicator(inputs, expecteds = None):
    model.zero_grad()

    inputs_vector = autograd.Variable(torch.FloatTensor([inputs]))

    log_probs = model(inputs_vector)

    if(expecteds != None):
        target = autograd.Variable(torch.LongTensor(expecteds))
        loss = loss_function(log_probs, target)
        loss.backward()
        optimizer.step()

    return log_probs


for epoch in range(NUM_EPOCHS):
    for inputs, expecteds in data:
        replicator(inputs, expecteds)

for inputs, expecteds in test_data:
    log_probs = replicator(inputs)
    print(log_probs)

