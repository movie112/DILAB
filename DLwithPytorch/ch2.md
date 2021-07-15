# 2. 신경망의 구성요소

## 신경망(Neural Network) 
> - 입력변수들 관련된 변수 간의 관계에 대해 학습하는 알고리즘
> - 신경망 구축(정의)  
>   + torch.nn.Sequential class 사용: 네트워크를 인스턴스화 하는 동시에 원하는 신경망에 > 연산순서를 인수로 전달
>   + torch.nn.Module 클래스 사용: 더 복잡하고 강력, 매끄러움         
>                       
>  #### 1. nn.Sequential(순차형 신경망)
> >   - torch.nn 모듈에 정의된 연산을 사용
> >   - 신경망을 인스턴스화 하기 위해 torch.nn.Sequential 클래스에 순차적으로 연산들을 인수로 전달
> >     + nn.Linear(입력크기, 출력크기): 가중치벡터 생성(연산)
> >     + nn.ReLU(): 비션형 함수, 출력을 비선형 영역으로 변환
> >     + nn.Sigmoid(): 예측결과 얻기 위해 0-1 사이의 값 출력

```python
import torch
import torch.nn as nn
my_nn = nn.Sequential(                           
    nn.Linear(3,2), 
    nn.ReLU(),    
    nn.Linear(2,1), 
    nn.Sigmoid())     
```

> #### 2. nn.Module
> > - self:   
> > 클래스에서 정의되는 모든 메소드에 전달되는 임의의 첫 번째 인수   
> > 클래스의 인스턴스를 나타냄   
> > 클래스에서 정의한 속성 및 메소드에 접근하는데 사용 가능
> > - __init__():   
> > 파이썬 클래스들에서 예약된 메소드(생성자)   
> > 클래스의 객체가 인스턴스화될 때마다 --init--() 메소드 안에서 랩핑된 코드 실행   
> > 객체가 인스턴스화되면 모든 신경망 연산이 준비되도록 해줌
> > - forward(self, x):   
> > 목표로 하는 레이어 연산을 통해 데이터의 흐름을 설명해야 함

```python
import torch.nn.functional as F
class MyNeuralNet(nn.Module):
    def __init__(self, input_size, n_nodes, output_size):
        super(MyNeuralNet, self).__init__()
        self.operationOne = nn.Linear(input_size, n_nodes)
        self.operationTwo = nn.Linear(n_nodes, output_size)  
        
    def forward(self, x):   # forward 메소드 정의
        x = F.relu(self.operationOne(x))     # ReLU
        x = self.operationTwo(x)
        x = F.sigmoid(x)
        return x
```
```python
 my_network = MyNeuralNet(input_size = 3, n_nodes = 2, output_size = 1)
```

> >  self.operationOne(x): 데이터를 신경망의 첫 번째 연산으로 전달   
> >  신경망에 접근하려면, MYNeuralNet 클래스의 객체를 인스턴스화해야 함   
> >  my_network 변수로 신경망으로 접근   

<br/>

## 텐서(tensor)
> - 파이토치에서 계산을 수행하는 엔진
> - 데이터 컨테이너
>   - 스칼라: 0차원
>    - 벡터: 1차원 텐서
>    - 매트릭스: 2차원 텐서, 2개의 축   
                
```python
# 1차원
first_order_tensor = torch.tensor([1, 2, 3])
print(first_order_tensor)
# 2차원
second_order_tensor = torch.tensor([ [ 11, 22, 33 ],
 [ 21, 22, 23 ]
 ])
print(second_order_tensor)
print(second_order_tensor[0, 1])

# print
tensor([[11, 22, 33],
        [21, 22, 23]])
tensor(22)
```

> - 2차원: [ [1차원텐서], [1차원 텐서] ]   
>          **ex) [0, 1]:**  0: 1차원텐서의 index / 1: 1차원텐서 내 요소의 index  
> - 4차원: [ [3차원텐서], [3차원텐서], [3차원텐서]...] 
> - .rand()   
> 원하는 모양의 랜덤요소를 갖는 tansor     

```python
random_tensor = torch.rand([4, 2])
#print
tensor([[0.8849, 0.2496],
        [0.0906, 0.8850],
        [0.3370, 0.4994],
        [0.2537, 0.1573]])
```

> - .view() 메소드   
>   tensor의 모양 변경, 요소들을 다른 축으로 옮김   
>   -1을 사용하면 Pytorch가 하나의 특정 축의 크기 계산

```python
random_tensor.view([2, 4])
#print
tensor([[0.8849, 0.2496, 0.0906, 0.8850],
        [0.3370, 0.4994, 0.2537, 0.1573]])
```
```python
random_tensor = torch.rand([4, 2, 4])
#print
tensor([[[0.4400, 0.6317, 0.8669, 0.9632],
         [0.5322, 0.7374, 0.7522, 0.7821]],

        [[0.5294, 0.0923, 0.5412, 0.9100],
         [0.6635, 0.9317, 0.3465, 0.2192]],

        [[0.8695, 0.5166, 0.2254, 0.4357],
         [0.2135, 0.7615, 0.7737, 0.1803]],

        [[0.2424, 0.2308, 0.3856, 0.8657],
         [0.3334, 0.9192, 0.7543, 0.5225]]])

random_tensor.view([2, -1, 4])
#print
tensor([[[0.4400, 0.6317, 0.8669, 0.9632],
         [0.5322, 0.7374, 0.7522, 0.7821],
         [0.5294, 0.0923, 0.5412, 0.9100],
         [0.6635, 0.9317, 0.3465, 0.2192]],

        [[0.8695, 0.5166, 0.2254, 0.4357],
         [0.2135, 0.7615, 0.7737, 0.1803],
         [0.2424, 0.2308, 0.3856, 0.8657],
         [0.3334, 0.9192, 0.7543, 0.5225]]])
```
<br/>

> - tensor의 연산   
>  연산 요소 별로 수행(+, -, *, /)

```python
torch.add(x, y)  # x + y 도 됨
torch.sub(x, y)  # x - y
torch.mul(x, y)  # x * y
torch.div(x, y)  # x / y
```

> &nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;행렬곱

```python
torch.matmul(x, y)  # x @ y 도 됨
```
> - tensor의 자료형

```python
x_float = torch.tensor([5, 3], dtype = torch.float32)
y_float = torch.tensor([3, 2], dtype = torch.float32)
print(x_float / y_float)
torch.FloatTensor([5, 3])
x.type(torch.DoubleTensor)
```

> - tensor로 dataset 가져오기

```python
# 1
import numpy as np
admit_data = np.genfromtxt('admit_status.csv', delimiter = ',', skip_header = 1)
print(admit_data)
# 2
admit_tensor = torch.from_numpy(admit_data)
```

## Pytorch에서 신경망 학습하기

> 앞서 정의한 신경망은 처음에 가중치를 랜덤 생성: 신경망에 바로 제이터를 전달하면, 의미없는 예측 결과 get
> - Criterion
>   - 참 값과 예측 값의 손실 스코어
> - Optimizer 
>   - 가능한한 신경망의 예측이 참 값에 가까워지도록 함 (가중치를 조정하기 위해)
> - 학습 단계(trianing phase)
>   - 신경망의 가중치를 업데이트하기 위해 Criterion의 loss를 이용하는 Opimizer의 반복적인 과정 
>   <br/>
> 신경망 학습 전에, 데이터셋을 입력x, 출력y로 분리
> 
```
x_train = admit_tensor[:300, 1:].float()
y_train = admit_tensor[:300, 0].float()
x_test = admit_tensor[300:, 1:].float()
y_test = admit_tensor[300:, 0].float()
```
> criterion, optimizer의 인스턴스 생성
>   - criterion
>       - torch.nn 모듈에 마련
>       - BCELoss(): 2진 분류, binary cross-entropy loss
>   
```
criterion = nn.BCELoss()
```

>   - optimizer   
>       - torch.optim 모듈에 내장
>       - 인자로 parameter or W -> parameters() 메소드를 이용하여 접근
>       - SGD optimizer: stochastic gradient descent optimizer

```
optimizer = torch.optim.SGD(my_network.parameters(), lr=0.01)
```

> 가중치를 업데이트하는 과정 for 루프
>   1. 예측 결과를 얻기 위해 데이터 전달: 신경망의 인스턴스의 인자로 입력 데이터 전달하면 됨
>   2. loss 계산: 예측과 참 값을 criterion에 전달하면 됨
>   3. 누적된 gradient 초기화: zero_grad() 메소드 이용
>   4. backpropagation 단계 수행: 계산된 loss의 backward() 메소드를 이용
>   5. W 업데이트: step() 메소드 

```
for epoch in range(100):
    # 1
    y_pred = my_network(x_train)
    # 2 
    loss_score = criterion(y_pred, y_train)
    print('epoch: ', epoch, 'loss: ', loss_score)
    
    # 3
    optimizer.zero_grad()
    # 4
    loss_score.backward()
    # 5
    optimizer.step()
```

<br/>

---
* 코드: <https://github.com/PacktPublishing/Deep-Learning-with-PyTorch-1.x/blob/master/Chapter02/Chapter02.ipynb>
