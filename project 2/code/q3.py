import torch
import matplotlib.pyplot as plt
import os

from data import create_dataloaders
from model import CNN

model = CNN()
train_loader, test_loader = create_dataloaders()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the weights
model.load_state_dict(torch.load("q1_model.pt", weights_only=True))
model.to(device)

model.eval()

os.makedirs("q3_filters", exist_ok=True)

# number of out channels in each layer
num_fs = [16, 32, 48, 64, 80]
norms = torch.zeros(len(test_loader.dataset), sum(num_fs))
all_labels = torch.zeros(len(test_loader.dataset))

step = 0
for images, labels in test_loader:
    images, labels = images.to(device), labels.to(device)
    # TODO: Forward pass with intermediate outputs = True
    _, [x1, x2, x3, x4, x5] =  model(images, intermediate_outputs=True)
    # TODO: Calculate the norm of the intermediate outputs along the
    # spatial dimensions. Check the PDF.

    x1 = torch.norm(x1, dim=(2, 3))
    x2 = torch.norm(x2, dim=(2, 3))
    x3 = torch.norm(x3, dim=(2, 3))
    x4 = torch.norm(x4, dim=(2, 3))
    x5 = torch.norm(x5, dim=(2, 3))

    # The following code assumes that x1, ..., x5 are the norms of the
    # intermediate outputs. You may need to change this to suit your code.

    f_idx = 0

    # Writing the norm values to the norms tensor
    norms[step:step+images.size(0), f_idx:f_idx+num_fs[0]] = x1.detach().cpu()
    f_idx += num_fs[0]
    norms[step:step+images.size(0), f_idx:f_idx+num_fs[1]] = x2.detach().cpu()
    f_idx += num_fs[1]
    norms[step:step+images.size(0), f_idx:f_idx+num_fs[2]] = x3.detach().cpu()
    f_idx += num_fs[2]
    norms[step:step+images.size(0), f_idx:f_idx+num_fs[3]] = x4.detach().cpu()
    f_idx += num_fs[3]
    norms[step:step+images.size(0), f_idx:f_idx+num_fs[4]] = x5.detach().cpu()

    # Saving the labels
    all_labels[step:step+images.size(0)] = labels
    
    step += images.size(0)

labelnames = ["airplane", "automobile", "bird", "cat", "deer",
              "dog", "frog", "horse", "ship", "truck"]

classwise_score_avg = torch.zeros(10, sum(num_fs))
for l in range(10):
    classwise_score_avg[l] = norms[all_labels == l].mean(dim=0)

start = 0
for layer_idx, num_f in enumerate(num_fs):
    os.makedirs(f"q3_filters/classwise_avg_{layer_idx}", exist_ok=True)

    for f_idx in range(num_f):
        fig, ax = plt.subplots()
        data = classwise_score_avg[:, start+f_idx]
        data /= data.max()
        ax.bar(labelnames, data)
        ax.set_title(f"Filter {f_idx}")
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"q3_filters/classwise_avg_{layer_idx}/filter_{f_idx}.png")
        plt.close()

    start += num_f
