from .sine_kan2 import SineKANLayer, SineKAN
from torchvision import datasets, transforms
import torch, numpy as np, matplotlib.pyplot as plt, torch.nn.functional as F
from torch import nn
from torch.profiler import profile, record_function, ProfilerActivity
# %% MNIST dataset
batch = 100
trainset = datasets.MNIST("./Data/mnist", train=True, download=True,
                                      transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True)

testset = datasets.MNIST(root='./Data/mnist', train=False, download=True, 
                         transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=False)
# %% Fashion MNIST
batch = 100
trainset = datasets.FashionMNIST('./Data/fashion-mnist', True, download=True,
                                 transform=transforms.ToTensor())
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch,
                                          shuffle=True)

testset = datasets.FashionMNIST('./Data/fashion-mnist', False, download=True, 
                                transform=transforms.ToTensor())
testloader = torch.utils.data.DataLoader(testset, batch_size=batch,
                                         shuffle=False)
# %%
device = torch.device('cuda:0')
net = nn.Sequential(
    nn.Flatten(),
    SineKAN([28*28, 128, 10])).to(device)
# %%
opt = torch.optim.AdamW(net.parameters(), 3e-4)
criterion = torch.nn.CrossEntropyLoss()
metric = lambda out, labels: (torch.argmax(out,1) == labels).float().mean()
# %% Profiling memory
with torch.profiler.profile(
        schedule=torch.profiler.schedule(wait=1, warmup=1, active=5, repeat=1),
        on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/sinekan2'),
        record_shapes=True,
        profile_memory=True,
        with_stack=True
) as prof:

    for i, data in enumerate(trainloader, 1):
        prof.step()  # Need to call this at each step to notify profiler of steps' boundary.
        if i >= 1 + 1 + 5:
            break
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = data
        inputs = inputs.to(device)
        labels = labels.to(device)
        
        # forward + backward + optimize
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
# %% Profile inference times
print("Trainable phases")
def profile_model(model, inputs, num_runs=100):
    labels = torch.randint(0, 10, (batch,)).to(device)
    times = []
    opt = torch.optim.AdamW(model.parameters(), 0.001)
    for r in range(num_runs):
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=True) as prof:
            opt.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            opt.step()

        # Extract total CUDA time spent on 'model_inference'
        time_total = 0.
        for event in prof.key_averages():
            time_total += event.cuda_time_total
        times.append(time_total)

    average_time = np.mean(times)
    std_time = np.std(times)
    print(f"Inference+Backprop time: {average_time} pm {std_time} ms")
     

inputs = torch.randn((batch, 5)).to(device)
in_dims = [10, 100]
Gs = [8, 16, 32]

for in_dim in in_dims:
    inputs = torch.randn((batch, in_dim)).to(device)
    for G in Gs:
        model = SineKAN(layers_hidden=[in_dim, 10], grid_size=G).to(device)
        print(f'Input dims: {in_dim}, Grid size: {G}')
        profile_model(model, inputs, 1000)
# %%
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
    print(f"Loss: {test_loss}, Accuracy: {test_acc}")
    return test_loss, test_acc

def train_loop(net, trainloader, testloader, epochs, opt):
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
        # if (ep+1)%5 == 0:
        #     print(f"Epoch: {ep+1}, Loss: {running_loss/i}, Accuracy: {running_acc/i}")
        history["epoch"].append(ep)
        history["train_loss"].append(running_loss / i)
        history["train_acc"].append(running_acc / i)
        
        test_loss, test_acc = test_fn(net, testloader)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
    return history

history = train_loop(net, trainloader, testloader, 30, opt)
# %%
import matplotlib.pyplot as plt

plt.plot([0] + history["epoch"], [0.9] + history["train_acc"])
plt.plot([0] + history["epoch"], [0.9] + history["test_acc"])
# %%
# %% Single layer regression
device = torch.device('cuda:0')
op = SineKANLayer(1, 1, grid_size=10).to(device)
# %%
opt = torch.optim.Adam(op.parameters(), lr=0.005)
criterion = torch.nn.MSELoss()
w = torch.arange(0.1, 2, 0.1).reshape(1, 1, -1).to(device)

epochs, batch = 4000, 100
for ep in range(epochs):
    x = (torch.rand(batch, 1).to(device)*6 - 3)
    y0 = ((2-w)*torch.sin(w*x.unsqueeze(-1))).sum(-1)
    
    opt.zero_grad()
    y = op(x)
    loss = criterion(y, y0)
    loss.backward()
    opt.step()
    print(f"Epoch: {ep}, MSE_loss: {loss.item()}")    
# %%
with torch.no_grad():
    for s in torch.arange(-3., 3., 1.):
        print(f"{s} to {s+1}")
        x = torch.linspace(s, s+1, 1000).to(device).unsqueeze(-1)
        y0 = ((2-w)*torch.sin(w*x.unsqueeze(-1))).sum(-1)
        y = op(x)
        plt.plot(x.cpu(), y0.cpu())
        plt.plot(x.cpu(), y.cpu())
        plt.grid()
        plt.show()

# %%