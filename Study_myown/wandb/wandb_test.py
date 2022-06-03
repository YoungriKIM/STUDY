import wandb
wandb.init(project="test-project", entity="ntdev")

import torch
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader

# dataset
transform = transforms.Compose(
    [transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

trainset = torchvision.datasets.CIFAR10(root='cifar_dataset', train=True, download=True, transform = transform)
trainloader = DataLoader(trainset, batch_size=8, shuffle = True, num_workers=2)
testset = torchvision.datasets.CIFAR10(root='cifar_dataset', train = False, download=True, transform=transform)
testloader = DataLoader(testset, batch_size=8, shuffle=True, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat','deer','dog','frog','horse','ship','truck')

# model
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16*5*5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)
    
    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16*5*5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
net = Net()

# 모델 확인
print(net)

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']
    
import torch.optim as optim
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

# 실직적인 학습 부분
# wandb.watch(net)
for epoch in range(8) :                                                 # epoch 횟수 설정

    running_loss = 0.0
    for i, data in enumerate(trainloader, 0):                           # 트레인로더에서 트레인 데이터 불러오기
        inputs, labels = data                                           # 데이터 안에는 이미지, 라벨이 있음
        optimizer.zero_grad()                                           # 옵티마이저 초기화 작업
        outputs = net(inputs)                                           # 이미지를 net 모델에 넣기
        loss = criterion(outputs, labels)                               # outputs과 labels을 비교해서 loss 계산
        loss.backward()                                                 # loss를 기준으로 gradient를 계산해서 backward를 진행
        optimizer.step()                                                # 이렇게 한 번 돌면 배치크기 만큼 돌리면서 loss를 누적하게 됨

        running_loss += loss.item()
        if i % 2000 == 1000 :                                           # 설정한 값 마다 loss 출력
            print('[%d, %5d] loss: %.3f' % (epoch + 1, i + 1, running_loss / 2000))
            wandb.log({'train_loss':running_loss/2000, 'lr':get_lr(optimizer)})
            running_loss = 0.0

    # test
    net.eval()
    correct = 0
    total = 0
    with torch.no_grad():   # 업데이트를 안 하는 것으로 설정
        for data in testloader:
            images, labels = data
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)   # == argmax / 10개의 라벨이 1개로 바뀐다
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print('1000개의 test 이미지에 대한 모델 가중치 정확도: %d %%'%(100 * correct / total))
    wandb.log({'val_acc':(100*correct/total)})
    net.train()
    
print('Finished Training')


