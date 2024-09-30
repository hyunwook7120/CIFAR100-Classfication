import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

class SwishActivation(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

# Squeeze-and-Excitation Block
# reduction_ratio : channel을 줄이는 비율, 일반적으로 4나 16사용
# in_channels : SEBlock에 입력되는 특성 맵의 채널 수를 나타
class SEBlock(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(SEBlock, self).__init__()
        # 전역 평균 풀링으로 특성 압축 -> 각 채널을 하나의 숫자로 압축
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        # 완전 연결 레이어로 채널의 중요도를 계산하는 부분
        self.excitation = nn.Sequential(
            # nn layer, 완전 연결 레이어 2개 사용
            # 채널 수를 줄임
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(inplace=True),
            #채널 수를 원래대로 복원
            nn.Linear(in_channels // reduction_ratio, in_channels),
            #각 채널의 중요도를 0과 1사이 값으로 규정
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.squeeze(x).view(b, c) # squeeze 단계
        y = self.excitation(y).view(b, c, 1, 1) # 중요도 계산(excitation) 단계
        return x * y.expand_as(x)

class MBConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion_factor, stride, k):
        super(MBConvBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.k = k

        exp_channels = in_channels * expansion_factor

        self.conv_layers = nn.Sequential(
            # expansion 1x1 conv를 사용하여 입력 채널 수를 expansion_factor 만큼 증가
            nn.Conv2d(in_channels, exp_channels, 1, bias=False),
            # conv 연산 후에 batch 정규화와 swish 활성화 함수를 적용(안정성 증가, 비선형성 추가)
            nn.BatchNorm2d(exp_channels),
            SwishActivation(),
            # Depthwise Separable Convolution, 파라미터 수 줄이면서 특성 추출할 수 있게 함
            nn.Conv2d(exp_channels, exp_channels, k, stride, k//2, groups=exp_channels, bias=False),
            nn.BatchNorm2d(exp_channels),
            SwishActivation(),
            # SEBlock
            SEBlock(exp_channels),
            # Projection: 채널 수 다시 줄임
            nn.Conv2d(exp_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

        # skip connection : 입력과 출력의 채널 수가 같고, stride 가 1일때 적용
        self.skip_connection = stride == 1 and in_channels == out_channels

    def forward(self, x):
        out = self.conv_layers(x)
        if self.skip_connection:
            out += x
        return out

class EfficientNet(nn.Module):
    def __init__(self, num_classes=100):
        super(EfficientNet, self).__init__()
        # 3개 채널을 32채널로 확장, 특성 맵의 공간 크기 줄이기
        self.initial_conv = nn.Sequential(
            nn.Conv2d(3, 32, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(32),
            SwishActivation()
        )

        self.blocks = nn.Sequential(
            MBConvBlock(32, 16, 1, 1, 3),
            MBConvBlock(16, 24, 6, 2, 3),
            MBConvBlock(24, 24, 6, 1, 3),
            MBConvBlock(24, 40, 6, 2, 5),
            MBConvBlock(40, 40, 6, 1, 5),
            MBConvBlock(40, 80, 6, 2, 3),
            MBConvBlock(80, 80, 6, 1, 3),
            MBConvBlock(80, 80, 6, 1, 3),
            MBConvBlock(80, 112, 6, 1, 5),
            MBConvBlock(112, 112, 6, 1, 5),
            MBConvBlock(112, 192, 6, 2, 5),
            MBConvBlock(192, 192, 6, 1, 5),
            MBConvBlock(192, 320, 6, 1, 3)
        )

        self.final_conv = nn.Sequential(
            nn.Conv2d(320, 1280, 1, bias=False),
            nn.BatchNorm2d(1280),
            SwishActivation()
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            # nn.Dropout(0.1),
            nn.Linear(1280, num_classes)
        )

    def forward(self, x):
        x = self.initial_conv(x)
        x = self.blocks(x)
        x = self.final_conv(x)
        x = self.classifier(x)
        return x