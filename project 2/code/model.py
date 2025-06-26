import torch
import torch.nn as nn

torch.manual_seed(123)

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding=3, stride=1, bias=False) # Conv 1 layer
        self.bn1 =  nn.BatchNorm2d(16) # BatchNorm layer
        self.conv2 = nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1, stride=1,  bias=False) # Conv 2 layer
        self.bn2 = nn.BatchNorm2d(32) # BatchNorm layer
        self.conv3 = nn.Conv2d(in_channels=32, out_channels=48, kernel_size=3, padding=1, stride=1,  bias=False) # Conv 3 layer
        self.bn3 = nn.BatchNorm2d(48) # BatchNorm layer
        self.conv4 = nn.Conv2d(in_channels=48, out_channels=64, kernel_size=3, padding=1, stride=1,  bias=False) # Conv 4 layer
        self.bn4 = nn.BatchNorm2d(64) # BatchNorm layer
        self.conv5 = nn.Conv2d(in_channels=64, out_channels=80, kernel_size=3, padding=1, stride=1) # Conv 5 layer
        self.relu = nn.ReLU() # ReLU layer
        self.maxpool = nn.MaxPool2d(kernel_size=2, stride=1, padding=1) # MaxPool layer
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)  # Avgpool layer

        # Dropout layers (added here)
        self.dropout1 = nn.Dropout(p=0.2)  # After conv4
        self.dropout2 = nn.Dropout(p=0.2)  # After conv5

        self.fc = nn.Linear(80, 10) # Linear layer
    
    def forward(self, x, intermediate_outputs=False):
        # TODO: Compute the forward pass output following the diagram in
        # the project PDF. If intermediate_outputs is True, return the
        # outputs of the convolutional layers as well.

        # First convolution block
        conv1_out = self.relu(self.bn1(self.conv1(x)))
        
        # Second convolution block
        conv2_out = self.relu(self.bn2(self.conv2(conv1_out)))

        # Third convolution block with Max Pooling
        conv3_out = self.relu(self.bn3(self.conv3(self.maxpool(conv2_out))))

        # Fourth convolution block with Max Pooling
        conv4_out = self.dropout1(self.relu(self.bn4(self.conv4(self.maxpool(conv3_out)))))

        # Fifth convolution block with Max Pooling
        conv5_out = self.dropout2(self.relu(self.conv5(self.maxpool(conv4_out))))

        # Adaptive Average Pooling
        pooled_out = self.avgpool(conv5_out)

        # Flattening before feeding into the fully connected layer
        flattened = torch.flatten(pooled_out, 1)

        # Fully connected layer
        final_out = self.fc(flattened)    #nn.functional.log_softmax(      , dim =-1 )

        if intermediate_outputs:
            return final_out, [conv1_out, conv2_out, conv3_out, conv4_out, conv5_out]
        else:
            return final_out
