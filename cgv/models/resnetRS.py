import torch
import torch.nn as nn
import torch.nn.functional as F

# Bottleneck block for ResNetRS
class Bottleneck(nn.Module):
    expansion = 4
    
    def __init__(self, in_planes, planes, stride=1, downsample=None, drop_rate=0.0):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.drop_rate = drop_rate
        self.dropout = nn.Dropout(drop_rate)
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        
        out = self.conv3(out)
        out = self.bn3(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        # Apply dropout if drop_rate > 0
        if self.drop_rate > 0:
            out = self.dropout(out)
        
        return out

# ResNetRS class
class ResNetRS(nn.Module):
    def __init__(self, block, layers, num_classes=1000, drop_rate=0.0):
        super(ResNetRS, self).__init__()
        self.in_planes = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        
        self.layer1 = self._make_layer(block, 64, layers[0], drop_rate=drop_rate)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, drop_rate=drop_rate)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2, drop_rate=drop_rate)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2, drop_rate=drop_rate)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)
    
    def _make_layer(self, block, planes, blocks, stride=1, drop_rate=0.0):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )
        
        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample, drop_rate))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes, drop_rate=drop_rate))
        
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x

# Function to build ResNetRS-50
def ResNetRS50(num_classes=100, drop_rate=0.0):
    return ResNetRS(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, drop_rate=drop_rate)



# CIFAR-100 Superclass 및 Class 매핑
CIFAR100_SUPERCLASS_MAPPING = {
    'aquatic mammals': [0, 1, 2, 3, 4],
    'fish': [5, 6, 7, 8, 9],
    'flowers': [10, 11, 12, 13, 14],
    'food containers': [15, 16, 17, 18, 19],
    'fruit and vegetables': [20, 21, 22, 23, 24],
    'household electrical devices': [25, 26, 27, 28, 29],
    'household furniture': [30, 31, 32, 33, 34],
    'insects': [35, 36, 37, 38, 39],
    'large carnivores': [40, 41, 42, 43, 44],
    'large man-made outdoor things': [45, 46, 47, 48, 49],
    'large natural outdoor scenes': [50, 51, 52, 53, 54],
    'large omnivores and herbivores': [55, 56, 57, 58, 59],
    'medium-sized mammals': [60, 61, 62, 63, 64],
    'non-insect invertebrates': [65, 66, 67, 68, 69],
    'people': [70, 71, 72, 73, 74],
    'reptiles': [75, 76, 77, 78, 79],
    'small mammals': [80, 81, 82, 83, 84],
    'trees': [85, 86, 87, 88, 89],
    'vehicles 1': [90, 91, 92, 93, 94],
    'vehicles 2': [95, 96, 97, 98, 99]
}

# 슈퍼클래스 예측 함수
def predict_superclass(output, mapping):
    superclass_scores = {}
    for superclass, class_indices in mapping.items():
        superclass_scores[superclass] = output[class_indices].sum()
    predicted_superclass = max(superclass_scores, key=superclass_scores.get)
    return predicted_superclass

# 클래스 예측 함수
def predict_class_within_superclass(output, mapping, predicted_superclass):
    class_indices = mapping[predicted_superclass]
    class_scores = output[class_indices]
    predicted_class = class_indices[class_scores.argmax()]
    return predicted_class
