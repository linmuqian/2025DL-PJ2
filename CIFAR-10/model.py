import torch.nn as nn
import torch.nn.functional as F
import torch

class Model1(nn.Module):
    def __init__(self):
        super(Model1, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x



class Model2(nn.Module):
    def __init__(self):
        super(Model2, self).__init__()
        
        # 1st convolutional layer
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(32)
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        # 2nd convolutional layer
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(64)
        
        # Fully connected layer (after flattening the conv layers)
        self.fc1 = nn.Linear(64 * 8 * 8, 512)  # The input size here depends on the image size after pooling
        self.fc2 = nn.Linear(512, 10)  # 10 output classes for CIFAR-10

        # Dropout layer for regularization
        self.dropout = nn.Dropout(p=0.5)

    def forward(self, x):
        # Convolutional Layer 1
        x = self.pool(F.relu(self.bn1(self.conv1(x))))
        
        # Convolutional Layer 2
        x = self.pool(F.relu(self.bn2(self.conv2(x))))
        
        # Flatten the tensor before feeding it into fully connected layers
        x = x.view(-1, 64 * 8 * 8)
        
        # Fully connected layer 1
        x = F.relu(self.fc1(x))
        
        # Apply dropout for regularization
        x = self.dropout(x)
        
        # Fully connected layer 2 (output layer)
        x = self.fc2(x)
        
        return x




class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        
        # First convolutional layer
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        
        # Second convolutional layer
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        # Skip connection (identity mapping)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)  # Add the shortcut (skip connection)
        out = F.relu(out)
        return out


class model_ResNet18(nn.Module):
    def __init__(self, num_classes=10):
        super(model_ResNet18, self).__init__()
        
        # Initial Convolution Layer
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        # Residual Blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        # Fully Connected Layer
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Initial Conv and BatchNorm
        x = self.maxpool(x)  # Maxpool after the first convolution
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        # Fully Connected Layer
        x = self.fc(x)
        
        return x


class model_ResNet18_C(nn.Module):
    def __init__(self, num_classes=10):
        super(model_ResNet18_C, self).__init__()
        
        # Adjusted initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Reduced kernel and stride
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual Blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.dropout = nn.Dropout(0.2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 256)  
        self.fc2 = nn.Linear(256, num_classes)  

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Initial Conv and BatchNorm
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)           # Apply dropout
        x = self.fc1(x)               # First FC layer
        x = F.gelu(x)                 # GeLU activation
        x = self.dropout(x)           # Another dropout for regularization
        x = self.fc2(x)               # Final classification layer
        
        return x
    


class model_ResNet18_C(nn.Module):
    def __init__(self, num_classes=10):
        super(model_ResNet18_C, self).__init__()
        
        # Adjusted initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Reduced kernel and stride
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual Blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.dropout = nn.Dropout(0.2)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 256)  
        self.fc2 = nn.Linear(256, num_classes)  

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Initial Conv and BatchNorm
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)           # Apply dropout
        x = self.fc1(x)               # First FC layer
        x = F.gelu(x)                 # GeLU activation
        x = self.dropout(x)           # Another dropout for regularization
        x = self.fc2(x)               # Final classification layer
        
        return x
    


class model_ResNet18_H(nn.Module):
    def __init__(self, num_classes=10):
        super(model_ResNet18_H, self).__init__()
        
        # Adjusted initial convolution
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)  # Reduced kernel and stride
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual Blocks
        self.layer1 = self._make_layer(64, 64, 2, stride=1)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        self.layer4 = self._make_layer(256, 512, 2, stride=2)
        
        self.dropout = nn.Dropout(0.3)
        
        # Fully Connected Layers
        self.fc1 = nn.Linear(512, 128)  
        self.fc2 = nn.Linear(128, 32)  
        self.fc3 = nn.Linear(32, num_classes)  

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Initial Conv and BatchNorm
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global Average Pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)           # Apply dropout
        x = self.fc1(x)               # First FC layer
        x = F.gelu(x)                 # GeLU activation
        x = self.dropout(x)           # Another dropout for regularization
        x = self.fc2(x)               # Secomd FC layer
        x = F.leaky_relu(x, negative_slope=0.01) # LeakyReLu activation
        x = self.fc3(x)               # Final classification layer

        return x
    

class model_ResNet34_C(nn.Module):
    def __init__(self, num_classes=10):
        super(model_ResNet34_C, self).__init__()
        
        # Initial convolution layer with smaller kernel size and stride
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)

        # Residual layers - ResNet34 uses more blocks than ResNet18
        self.layer1 = self._make_layer(64, 64, 3, stride=1)  # 3 blocks
        self.layer2 = self._make_layer(64, 128, 4, stride=2)  # 4 blocks
        self.layer3 = self._make_layer(128, 256, 6, stride=2)  # 6 blocks
        self.layer4 = self._make_layer(256, 512, 3, stride=2)  # 3 blocks
        
        self.dropout = nn.Dropout(0.2)
        
        # Fully connected layers for classification
        self.fc1 = nn.Linear(512, 256)  
        self.fc2 = nn.Linear(256, num_classes)  

    def _make_layer(self, in_channels, out_channels, blocks, stride):
        layers = []
        layers.append(BasicBlock(in_channels, out_channels, stride))
        for _ in range(1, blocks):
            layers.append(BasicBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))  # Initial convolution and batch normalization
        
        # Residual layers
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        # Global average pooling
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = torch.flatten(x, 1)
        
        x = self.dropout(x)           # Apply dropout for regularization
        x = self.fc1(x)               # First fully connected layer
        x = F.gelu(x)                 # GeLU activation function
        x = self.dropout(x)           # Another dropout layer
        x = self.fc2(x)               # Final classification layer
        
        return x    
    

class _DenseLayer(nn.Module):
    def __init__(self, num_input_features, growth_rate, bn_size, drop_rate, activation='GELU'):
        super(_DenseLayer, self).__init__()
        self.norm1 = nn.BatchNorm2d(num_input_features)
        self.activation1 = nn.GELU() if activation == 'GELU' else nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(num_input_features, bn_size * growth_rate, kernel_size=1, stride=1, bias=False)
        self.norm2 = nn.BatchNorm2d(bn_size * growth_rate)
        self.activation2 = nn.GELU() if activation == 'GELU' else nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(bn_size * growth_rate, growth_rate, kernel_size=3, stride=1, padding=1, bias=False)
        self.drop_rate = drop_rate

    def forward(self, x):
        if isinstance(x, torch.Tensor):
            prev_features = [x]
        else:
            prev_features = x
        concated_features = torch.cat(prev_features, 1)
        bottleneck = self.conv1(self.activation1(self.norm1(concated_features)))
        new_features = self.conv2(self.activation2(self.norm2(bottleneck)))
        if self.drop_rate > 0:
            new_features = F.dropout(new_features, p=self.drop_rate, training=self.training)
        return new_features

class _DenseBlock(nn.ModuleDict):
    def __init__(self, num_layers, num_input_features, bn_size, growth_rate, drop_rate, activation='GELU'):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                activation=activation
            )
            self.add_module(f'denselayer{i+1}', layer)

    def forward(self, init_features):
        features = [init_features]
        for layer in self.values():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)

class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features, activation='GELU'):
        super(_Transition, self).__init__()
        self.norm = nn.BatchNorm2d(num_input_features)
        self.activation = nn.GELU() if activation == 'GELU' else nn.ReLU(inplace=True)
        self.conv = nn.Conv2d(num_input_features, num_output_features, kernel_size=1, stride=1, bias=False)
        self.pool = nn.AvgPool2d(kernel_size=2, stride=2)

class model_DenseNet(nn.Module):
    def __init__(self, growth_rate=24, block_config=[6, 12, 48, 32], 
                 num_init_features=64, bn_size=4, drop_rate=0.2,
                 num_classes=10, in_channels=3, activation='GELU', compression=2):
        super(model_DenseNet, self).__init__()
        self.activation = activation
        
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, num_init_features, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(num_init_features),
            nn.GELU() if activation == 'GELU' else nn.ReLU(inplace=True)
        )
        
        # DenseBlock + Transition
        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                activation=activation
            )
            self.features.add_module(f'denseblock{i+1}', block)
            num_features += num_layers * growth_rate
            if i != len(block_config) - 1:
                out_features = num_features // compression
                trans = _Transition(num_input_features=num_features, num_output_features=out_features, activation=activation)
                self.features.add_module(f'transition{i+1}', trans)
                num_features = out_features
        
        self.features.add_module('norm_final', nn.BatchNorm2d(num_features))
        self.features.add_module('activation_final', nn.GELU() if activation == 'GELU' else nn.ReLU(inplace=True))
        self.features.add_module('avg_pool', nn.AdaptiveAvgPool2d((1, 1)))
        

        self.classifier = nn.Sequential(
            nn.Dropout(drop_rate),
            nn.Linear(num_features, 256),
            nn.GELU() if activation == 'GELU' else nn.ReLU(inplace=True),
            nn.Dropout(drop_rate),
            nn.Linear(256, num_classes)
        )
        

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x