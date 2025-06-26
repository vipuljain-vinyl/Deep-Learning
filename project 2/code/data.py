import torch
from torch.utils.data import DataLoader
from torchvision.datasets import ImageFolder
from torchvision.transforms import v2

torch.manual_seed(123)


def create_dataloaders(batch_size=32):


    train_tf = v2.Compose([
        #v2.RandomResizedCrop(size=224, scale=(0.8, 1.0)),
        v2.RandomHorizontalFlip(),
        # v2.RandomRotation(15),
        v2.ToImage(),  # ensures input is in image format
        v2.ToDtype(torch.float32, scale=True),  # converts to float and scales to [0,1]
        v2.Normalize(mean=[0.438, 0.435, 0.422], std=[0.228, 0.225, 0.231]),
        
        v2.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3)),  # Randomly erase part of image
        v2.GaussianNoise(mean =0.0, sigma=0.01, clip=False),  # Built-in Gaussian noise for robustness
        
         ])


    val_tf = v2.Compose([
        # v2.Resize(256),
        # v2.CenterCrop(224),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.438, 0.435, 0.422], std=[0.228, 0.225, 0.231])
    ]) # TODO: Define the validation transform. No random augmentations here.

    train_dataset =  ImageFolder(root='custom_image_dataset/train', transform=train_tf) # TODO: Load the train dataset. Make sure to pass train_tf to it.
    val_dataset = ImageFolder(root='custom_image_dataset/val', transform=val_tf) # TODO: Load the val dataset.

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=1) # TODO: Create the train dataloader
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=1) # TODO: Create the val dataloader

    return train_loader, val_loader

