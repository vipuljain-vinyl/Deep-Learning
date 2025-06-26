import torch
import imageio.v2 as imio
import os

from model import CNN

model = CNN()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# load the weights
model.load_state_dict(torch.load("q1_model.pt", weights_only=True))

model.eval()

conv_weights = model.conv1.weight.data.clone().cpu() # Get the conv1 layer weights

os.makedirs("q2_filters", exist_ok=True)

for i in range(conv_weights.shape[0]):
    f = conv_weights[i]  # get the i-th filter
    # TODO: Normalize the filter to [0, 255] as convert it to uint8.

    # Normalize the filter to [0, 1] per filter
    f_min = f.min()
    f_max = f.max()
    f_norm = (f - f_min) / (f_max - f_min + 1e-5)
    
    # Convert to numpy and scale to [0, 255]
    f_uint8 = (f_norm.numpy().transpose(1, 2, 0) * 255).astype("uint8")
    
    # Otherwise, it will not be visualized correctly.
    imio.imwrite(f"q2_filters/filter_{i}.png", f_uint8)
