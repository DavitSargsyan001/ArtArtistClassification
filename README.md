# ArtArtistClassification

## Project Overview
This project is a University class project developed collaboratively with fellow classmates. The objective of the project is to classify artworks using machine learning techniques. Specifically, we utilized a convolutional neural network (CNN) to distinguish between different styles and artists. 

## Dataset
We used the Impressionist Classifier Dataset, which can be found [here](https://www.kaggle.com/datasets/delayedkarma/impressionist-classifier-data?resource=download). For your convenience, the dataset is also uploaded in this repository.

## Neural Network
We employed ResNet-18, a variant of the ResNet (Residual Network) architecture. ResNet-18 is known for its efficiency and effectiveness in image classification tasks due to its use of residual connections, which help in training deeper networks by mitigating the vanishing gradient problem.

## Code Explanation
The code provided demonstrates the implementation of the art classification model using Python and PyTorch. Below is an overview of the code:

### 1. Import Libraries
The necessary libraries and modules are imported, including PyTorch, torchvision for data handling and model loading, and other utility modules.

```python
import numpy as np
import pandas as pd
import os
import torch
import torchvision
from torchvision import datasets, models, transforms
import torch.utils.data as data
from torch.utils.tensorboard import SummaryWriter
import torch.nn as nn
import torch.optim as optim
from torch.optim import lr_scheduler
import time, os, copy, argparse
import multiprocessing
from matplotlib import pyplot as plt
from torchvision import transforms

```

## Data Loading and Transformation

Data directories are defined, and image transformations are applied to augment the training data and normalize the images.

```python
train_directory = '/archive/training/training'
valid_directory = '/archive/validation/validation'
    
image_transforms = { 
    'train': transforms.Compose([
        transforms.RandomResizedCrop(size=256, scale=(0.8, 1.0)),
        transforms.RandomRotation(degrees=15),
        transforms.RandomHorizontalFlip(),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ]),
    'valid': transforms.Compose([
        transforms.Resize(size=256),
        transforms.CenterCrop(size=224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406],
                             [0.229, 0.224, 0.225])
    ])
}

dataset = {
    'train': datasets.ImageFolder(root=train_directory, transform=image_transforms['train']),
    'valid': datasets.ImageFolder(root=valid_directory, transform=image_transforms['valid'])
}

dataset_sizes = {
    'train': len(dataset['train']),
    'valid': len(dataset['valid'])
}

dataloaders = {
    'train': data.DataLoader(dataset['train'], batch_size=64, shuffle=True, pin_memory=True, drop_last=True),
    'valid': data.DataLoader(dataset['valid'], batch_size=64, shuffle=True, pin_memory=True, drop_last=True)
}

class_names = dataset['train'].classes
```

## Model Setup

The ResNet-18 model is loaded and modified to fit the number of classes in the dataset. The model is then moved to the GPU if available.

```python
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model_ft = models.resnet18(pretrained=True)
num_ftrs = model_ft.fc.in_features
model_ft.fc = nn.Linear(num_ftrs, len(class_names))
model_ft = model_ft.to(device)
```

## Training the Model

The training function is defined, including the training loop, validation, and logging of performance metrics using TensorBoard.

```python
criterion = nn.CrossEntropyLoss()
optimizer_ft = optim.SGD(model_ft.parameters(), lr=0.001, momentum=0.9)
exp_lr_scheduler = lr_scheduler.StepLR(optimizer_ft, step_size=7, gamma=0.1)

def train_model(model, criterion, optimizer, scheduler, num_epochs=30):
    since = time.time()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    writer = SummaryWriter()
    
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        for phase in ['train', 'valid']:
            if phase == 'train':
                model.train()
            else:
                model.eval()
            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                writer.add_scalar('Train/Loss', epoch_loss, epoch)
                writer.add_scalar('Train/Accuracy', epoch_acc, epoch)
                writer.flush()
            else:
                writer.add_scalar('Valid/Loss', epoch_loss, epoch)
                writer.add_scalar('Valid/Accuracy', epoch_acc, epoch)
                writer.flush()

            if phase == 'valid' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))
    model.load_state_dict(best_model_wts)
    return model

model_ft = train_model(model_ft, criterion, optimizer_ft, exp_lr_scheduler, num_epochs=30)
```

## Saving the Model

The trained model is saved for future use.

```python
PATH = "model_1.pth" 
print("\nSaving the model...")
torch.save(model_ft, PATH)
```

## How to use

1. Clone the repository.
2. Download and extract the dataset from the provided Kaggle link.
3. Adjust the train_directory and valid_directory paths to point to the dataset locations.
4. Run the script to train the model.
5. The trained model will be saved as model_1.pth.

## Future Improvements

. Experiment with different neural network architectures.
. Hyperparameter tuning for better performance.
. Implementation of more advanced data augmentation techniques.

Feel free to contribute to this project by forking the repository and submitting pull requests.

## Acknowledgements

We acknowledge the use of the Impressionist Classifier Dataset from Kaggle and the PyTorch framework for developing this project.
