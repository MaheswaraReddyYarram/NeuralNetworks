import torch
import torch.nn as nn

if __name__ == '__main__':
    print(torch.__version__)
    model = nn.Linear(10,3)
    loss = nn.MSELoss()
    ## dummy input x
    input_vector = torch.randn(10)
    ## class number 3, denoted as a vector with the class index to 1
    target = torch.tensor([0,0,1])
    ## y in math
    pred = model(input_vector)
    output = loss(pred, target)
    print("Prediction: " ,pred)
    print("Output: " , output)