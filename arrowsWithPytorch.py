from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader, TensorDataset

from torchvision import datasets
from torchvision.transforms import ToTensor

import torch

from arrowRecognitonShallow import data

class ImageClassifier(nn.Module): 
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3,32,(3,3)),
            nn.ReLU(),
            nn.Conv2d(32,64,(3,3)),
            nn.ReLU(),
            nn.Conv2d(64,1,(3,3)),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(25*25,10),
            nn.ReLU(),
            nn.Linear(10,1))

            # input channel is 1... image black and white
            # 32 3x3 filters

            # flatten will turn set off 64 outputs into a tensors


    def forward(self,x):
        return self.model(x)



clf = ImageClassifier().to("cpu")
opt = Adam(clf.parameters(),lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

# training flow
numExamples = 5000
dataGet = data(numExamples,True)
X_train = torch.from_numpy(dataGet.x).float()
y_train = torch.from_numpy(dataGet.y.T).float()

dataset = TensorDataset(X_train, y_train)

batch_size = 32  # You can adjust this batch size as needed
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

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

    with open("arrowModelState.pt", "wb") as f:
        save(clf.state_dict(),f)





    
