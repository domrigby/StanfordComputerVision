from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

import torch

# get data
train = datasets.MNIST(root="data",download=True,train=True,transform=ToTensor())
dataset = DataLoader(train,32) # batches of 32 images

class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(1,32,(3,3)),
            nn.ReLU(),
            nn.Conv2d(32,64,(3,3)),
            nn.ReLU(),
            nn.Conv2d(64,64,(3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64*(28-6)*(28-6),10))

            # input channel is 1... image black and white
            # 32 3x3 filters

            # flatten will turn set off 64 outputs into a tensors


    def forward(self,x):
        return self.model(x)



clf = ImageClassifier().to("cpu")
opt = Adam(clf.parameters(),lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# training flow

if __name__ == "__main__":
    for epoch in range(10):
        for batch in dataset:
            X,y = batch 
            yhat = clf(X)
            loss = loss_fn(yhat,y)

            opt.zero_grad()
            loss.backward()

            opt.step()
        
        print(f"Epoch {epoch} Loss: {loss.item()}")

    with open("modelState.pt", "wb") as f:
        save(clf.state_dict(),f)
    
