import torch
import torch.nn as nn
from icecream import ic

def train():
    model = nn.Linear(4, 2)
    loss_fn = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for epoch in range(10):
        # convert inputs to tensors
        inputs = torch.Tensor([0.8, 0.4, 0,4, 0.2])
        targets = torch.Tensor([1, 0])
        #clear gradient buffers
        optimizer.zero_grad()

        # get output from the model, given the inputs
        outputs = model(inputs)

        # get loss for the predicted output
        loss = loss_fn(outputs, targets)
        ic(loss)

        # get gradients w.r.t to parameters
        loss.backward()

        #update parameters
        optimizer.step()

        ic('epoch:{}, loss:{}', epoch, loss)


if __name__ == '__main__':
    print(torch.__version__)
    model = nn.Linear(10,3)
    ic(model.weight.shape, model.bias.shape, model.parameters())
    loss = nn.MSELoss()
    ## dummy input x
    input_vector = torch.randn(10)
    ic(input_vector)
    ## class number 3, denoted as a vector with the class index to 1
    target = torch.tensor([0,0,1])
    ## y in math
    pred = model(input_vector)
    output = loss(pred, target)
    print("Prediction: " ,pred)
    print("Output: " , output)

    train()