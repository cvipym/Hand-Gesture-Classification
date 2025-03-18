import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

NUM_CLASSES = 10

#모델 아키텍처 정의 ( 학습 시 사용한 구조와 동일하게. )

class KeyPointClassifier(nn.Module):
    def __init__(self, input_size=42, num_classes=10):
        super(KeyPointClassifier,self).__init__()
        print('mark: basic classifier on')
        self.dropout_input = nn.Dropout(p=0.2)
        self.fc1 = nn.Linear(input_size, 20)
        self.dropout1 = nn.Dropout(p=0.4)
        self.fc2 = nn.Linear(20, 10)
        self.fc3 = nn.Linear(10, num_classes)
        self.log_softmax = nn.LogSoftmax(dim=1)
        print('mark: basic init done')
        
        
    def forward(self,x):
        x = self.dropout_input(x)
        x = F.relu(self.fc1(x))
        x = self.dropout1(x)
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.log_softmax(x)
        return x