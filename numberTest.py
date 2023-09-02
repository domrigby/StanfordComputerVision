import torch 
import matplotlib.pyplot as plt

from MNISTPytorch import ImageClassifier

from torch import nn, save, load
from torch.optim import Adam
from torch.utils.data import DataLoader

from torchvision import datasets
from torchvision.transforms import ToTensor

import random

clf = ImageClassifier().to("cpu")
opt = Adam(clf.parameters(),lr=1e-3)
loss_fn = nn.CrossEntropyLoss()

train = datasets.MNIST(root="data",download=True,train=True,transform=ToTensor())

while True:
    with open("modelState.pt","rb") as f:
            clf.load_state_dict(load(f))

    idx = random.randint(0, len(train)-1)
    image, label =train[idx]

    # Get model prediction
    with torch.no_grad():
        output = clf(image.unsqueeze(0))  # Add batch dimension

    predicted_class = torch.argmax(output, dim=1).item()

    # Display the image and prediction
    plt.figure(figsize=(6, 8))
    plt.subplot(211)
    plt.imshow(image.squeeze().numpy(), cmap='gray')
    plt.title(f"True Label: {label}")

    plt.subplot(212)
    plt.bar(range(10), torch.nn.functional.softmax(output[0], dim=0).numpy())
    print(torch.nn.functional.softmax(output[0], dim=0).numpy())
    plt.xlabel('Class')
    plt.ylabel('Probability')
    plt.title(f"Predicted Label: {predicted_class}")

    plt.tight_layout()
    plt.show()
