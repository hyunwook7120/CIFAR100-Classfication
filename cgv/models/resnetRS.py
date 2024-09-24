import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

# BasicBlock 정의
class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut = downsample

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.shortcut is not None:  # shortcut이 None인지 확인
            out += self.shortcut(x)  # shortcut 경로
        else:
            out += x  # shortcut이 None일 경우, 입력을 그대로 더함
        out = F.relu(out)
        return out

# ResNetRS 모델 정의
class ResNetRS(nn.Module):
    def __init__(self, block, num_blocks):
        super(ResNetRS, self).__init__()
        self.in_channels = 64
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, num_blocks[0])
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)  # stride=2 추가
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)  # stride=2 추가
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)  # stride=2 추가
        self.fc = nn.Linear(512, 100)  # CIFAR-100 클래스 수에 맞춤
        self.dropout = nn.Dropout(0.3) # 드롭아웃 비율 30%
        self.initialize_weights() # 가중치 초기화 호출

    def _make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))

        return nn.Sequential(*layers)
        
    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                init.xavier_uniform_(m.weight)  # Xavier 초기화
                if m.bias is not None:
                    init.zeros_(m.bias)  # 편향은 0으로 초기화

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.maxpool(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.adaptive_avg_pool2d(out, (1, 1))  # adaptive_avg_pool2d로 수정
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)  # softmax 활성화 함수 적용
        return out


# ResNetRS18 모델 생성 함수
def ResNetRS18():
    return ResNetRS(BasicBlock, [2, 2, 2, 2])

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
