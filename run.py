# %%
from sine_kan2 import SineKANLayer, SineKAN
from torchvision import datasets, transforms
import torch, numpy as np, matplotlib.pyplot as plt, torch.nn.functional as F
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
import itertools as it
# %% MNIST dataset
batch = 128
trainset = datasets.MNIST("./Data/mnist", train=True, download=True,
                                      transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True)

testset = datasets.MNIST(root='./Data/mnist', train=False, download=True, 
                         transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=False)
# %%
criterion = torch.nn.CrossEntropyLoss()
metric = lambda out, labels: (torch.argmax(out,1) == labels).float().mean()

def test_fn(net, testloader):       
    net.eval()
    
    running_loss = 0.
    running_acc = 0.
    for (i, data) in enumerate(testloader, 1):
        with torch.no_grad():
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            acc = metric(outputs, labels)
            
            running_acc += acc.item()
            running_loss += loss.item()
    test_loss, test_acc = running_loss/i, running_acc/i
    # print(f"Loss: {test_loss}, Accuracy: {test_acc}")
    return test_loss, test_acc

def train_loop(net, trainloader, testloader, epochs, opt, schedule):
    history = {"epoch": [],
               "train_loss": [],
               "train_acc": [],
               "test_loss": [],
               "test_acc": [],
               }
    
    print("Training...")
    net.train()
    for ep in range(epochs):
        running_loss = 0.
        running_acc = 0.
        for i, data in enumerate(trainloader, 1):
            # get the inputs; data is a list of [inputs, labels]
            inputs, labels = data
            inputs = inputs.to(device)
            labels = labels.to(device)

            # zero the parameter gradients
            opt.zero_grad()
            
            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()
            # Metrics and logging
            with torch.no_grad():
                acc = metric(outputs, labels)
            # update statistics
            running_acc += acc.item()
            running_loss += loss.item()
            
        schedule.step()
        history["epoch"].append(ep)
        history["train_loss"].append(running_loss / i)
        history["train_acc"].append(running_acc / i)
        
        test_loss, test_acc = test_fn(net, testloader)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
    return history
# %%
device = torch.device('cuda:0')
hparams = {
    "learning_rate": [1e-4, 3e-4, 7e-4, 1e-3],
    "weight_decay": [0.5, 0.1, 0.05],
    "lr_decay": [0.9, 0.5, 0.1],
    "phase_factor": [0.8, 0.5, 0.1],
}

hparam_hist = {
    "learning_rate": [],
    "weight_decay": [],
    "lr_decay": [],
    "phase_factor": [],
    "history": [],
}
# %%
for (lr, w_decay, lr_decay, phase_factor) in it.product(*hparams.values()):
    net = nn.Sequential(
        nn.Flatten(),
        SineKAN([28*28, 128, 10], phase_factor=phase_factor)
    ).to(device)
    
    opt = torch.optim.AdamW(net.parameters(), lr, weight_decay=w_decay)
    schedule = torch.optim.lr_scheduler.StepLR(opt, 5, lr_decay)
    
    history = train_loop(net, trainloader, testloader, 200, opt, schedule)

    hparam_hist["learning_rate"].append(lr)
    hparam_hist["weight_decay"].append(w_decay)
    hparam_hist["lr_decay"].append(lr_decay)
    hparam_hist["phase_factor"].append(phase_factor)
    hparam_hist["history"].append(history)
    
    print(f"lr={lr}, w_decay={w_decay}, lr_decay={lr_decay}, phase_factor={phase_factor}")
    best_acc = np.max(history["test_acc"])
    print(f"Best Test accuracy: {best_acc}")
    print("<----->")
# %%