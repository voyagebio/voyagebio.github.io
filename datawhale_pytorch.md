

# 第一章、参考文档及社区

1.Awesome-pytorch-list(https://github.com/bharathgs/Awesome-pytorch-list)

2.PyTorch官方文档(https://pytorch.org/docs/stable/index.html)

3.[Pytorch-handbook](https://github.com/zergtant/pytorch-handbook)

4.[PyTorch官方社区](https://discuss.pytorch.org/)



# 第二章

## 2.1 张量Tensor

核心：一个数据容器。可包含数字或者字符串

优势：提供GPU计算和自动求梯度等功能、



#### 创建tensor：

1.

```python
x = torch.rand(4, 3) #(个数，维度)
x = torch.tensor([5.5, 3]) #直接把【5.5,3】拿来用

#基于已经存在的tensor创建一个新的tensor
x = torch.ones(4, 3, dtype = double)
x2 = torch.randn_like(x, dtype=torch.float)
```

其他一些构造Tensor的函数

<img src="C:\Users\22454\AppData\Roaming\Typora\typora-user-images\image-20220516141035440.png" alt="image-20220516141035440" style="zoom:200%;" />



#### 张量操作

```python
#加法
result = torch.add(x,y)

#索引（结果与原数据共享内存，修改一个，另一个会跟着修改。如果不想修改，可以考虑使用copy()等方法）
print(x[:, 1]) #取第二列

#改变tensor的大小（仅仅更改观察角度，与原tensor共享内存，可以clone一个副本再view）
x = torch.randn(4, 4)
y = x.view(16)
z = x.view(-1, 8) # -1是指这一维的维数由其他维度决定，实际变为2*8维
```

注意点：当两个形状不同的Tensor按元素计算时，可能会触发广播（broadcast）机制：显示当复制元素使两个Tensor形状相同后再按元素计算。



### 2.2 自动求导

1.如果设置它的属性` .requires_grad` 为 `True`，那么它将会追踪对于该张量的所有操作。当完成计算后可以通过调用` .backward()`，来自动计算所有的梯度。这个张量的所有梯度将会自动累加到`.grad`属性

2.调用`.detach()`方法将其与计算历史分离，并阻止它未来的计算记录被跟踪。为了防止跟踪历史记录(和使用内存），可以将代码块包装在 `with torch.no_grad(): `中。在评估模型时特别有用，因为模型可能具有 `requires_grad = True` 的可训练的参数，但是我们不需要在此过程中对他们进行梯度计算。

3.每个张量都有一个`.grad_fn`属性，该属性引用了创建 `Tensor `自身的`Function`。如果张量是用户手动创建，那么这个张量的`grad_fn`是 `None` 

```python
#可以通过设置requires_grad=True来追踪其计算历史
#如果没有指定的话，默认requires_grad=Flase
x = torch.ones(2, 2, requires_grad=True)
```

4.反向传播求梯度（原理：链式求导法则）

```python
x = torch.ones(2, 2, requires_grad=True)
y = x**2
z = y * y * 3
out = z.mean()#均值
print(out)
#d(out)/dx
out.backward(torch.tensor(1.))
#因为 out 是一个标量，因此out.backward()和 out.backward(torch.tensor(1.)) 等价
print(x.grad)
```

Tip：grad在反向传播中是累加的，因此在进行下一次反向传播前要把梯度清零

```
x.grad.data.zero_()
```

5.阻止 autograd 跟踪设置了`.requires_grad=True`的张量的历史记录

```python
print((x ** 2).requires_grad)#ture
with torch.no_grad():
    print((x ** 2).requires_grad)#flase
```

6.想要修改 tensor 的数值，但是又不希望被 autograd 记录(即不会影响反向传播)

```python
#方法：对tensor.data进行操作
x = torch.ones(1,requires_grad=True)
#tensor([1.])
print(x.data) # 还是一个tensor
print(x.data.requires_grad) # 但是已经是独立于计算图之外

y = 2 * x
x.data *= 100 # 只改变了值，不会记录在计算图，所以不会影响梯度传播

y.backward()
print(x) # 更改data的值也会影响tensor的值，tensor([100.], requires_grad=True)
print(x.grad)#tensor([2.])
```



### 2.3 并行运算

##### 概念：让多个GPU一起训练

在编写程序中，当我们使用了 `cuda()` 时，其功能是让我们的模型或者数据迁移到GPU当中，通过GPU开始计算。

##### 方法：

1.网络结构分布到不同设备中：GPU的通信在这种密集任务中很难办到，因此不可

2.同一层的任务分布到不同数据中：需要大量的训练，同步任务加重的情况下，会出现和第一种方式一样的问题，因此不可

3.不同的数据分布到不同的设备中：可

ps：现在的主流方式是数据并行的方式(Data parallelism)



# 第三章

### 3.1步骤

step1：预处理，包括数据格式的统一和必要的数据变换，划分训练集和预测集
step2：选择模型，设定损失函数、优化函数、超参数
step3：用模型拟合

### 3.2基本配置

```python
batch_size = 16#每次读入的样本数
lr = 1e-4#初始学习率
max_epochs = 100#训练次数
#GPU配置
# 方案一：使用os.environ，这种情况如果使用GPU不需要设置
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# 方案二：使用“device”，后续对要使用GPU的变量用.to(device)即可
device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
```

### 3.3数据读入

Dataset定义好数据的格式和数据变换形式，DataLoader用iterative的方式不断读入批次数据。

需要定义自己的Dataset类（继承pytorch的Dataset）实现灵活的数据读入

```python
class MyDataset(Dataset):
    def __init__(self):
        #用于向类中传入外部参数，同时定义样本集 
    def __getitem__(self, index):
       #用于逐个读取样本集合中的元素，可以进行一定的变换，并将返回训练/验证所需的数据
    def __len__(self):
       #用于返回数据集的样本数
```

根据Dataset按批次读取数据

```python
from torch.utils.data import DataLoader
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=4, shuffle=True, drop_last=True)
val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size, num_workers=4, shuffle=False)
#num_workers：有多少个进程用于读取数据
#shuffle：是否将读入的数据打乱
#drop_last：对于样本最后一部分没有达到批次数的样本，使其不再参与训练
```

### 3.4模型构建

#### 3.4.1神经网络的构建（基本框架）

基于 Module 类的模型来完成神经网络构建，需要继承Module 类的 init 函数和 forward 函数

自定义的这个类是一个可供⾃由组建的部件。它的子类既可以是⼀个层(如PyTorch提供的 Linear 类)，⼜可以是一个模型(如这里定义的 MLP 类)，或者是模型的⼀个部分。

```python
class MLP(nn.Module):
  # 声明带有模型参数的层，这里声明了两个全连接层
  def __init__(self, **kwargs):
    # 调用MLP父类Block的构造函数来进行必要的初始化。这样在构造实例时还可以指定其他函数
    super(MLP, self).__init__(**kwargs)
    self.hidden = nn.Linear(784, 256)
    self.act = nn.ReLU()
    self.output = nn.Linear(256,10)
    #⽆须定义反向传播函数，系统将通过⾃动求梯度⽽自动⽣成反向传播所需的 backward 函数。
   # 定义模型的前向计算，即如何根据输入x计算返回所需要的模型输出
  def forward(self, x):
    o = self.act(self.hidden(x))
    return self.output(o)   
```

#### 3.4.2自定义层

1.不含模型参数的层

```python
#自定义了一个将输入减掉均值后输出的层，并将层的计算定义在了 forward 函数里
class MyLayer(nn.Module):
    def __init__(self, **kwargs):
        super(MyLayer, self).__init__(**kwargs)
    def forward(self, x):
        return x - x.mean()
```

2.含有模型参数的层

Parameter类：是该类的Tensor会被自动添加进模型的参数列表里

在自定义层时，应该把参数定义为Parameter类，还可以使⽤ ParameterList 和 ParameterDict 分别定义参数的列表和字典。

**二维卷积层：**将输入和卷积核做互相关运算，并加上一个标量偏差来得到输出。

**池化层：**每次对输入数据的一个固定形状窗口(⼜称池化窗口)中的元素计算输出，直接计算池化窗口内元素的最大值或者平均值

####3.4.3模型示例

典型训练过程：

1. 定义包含一些可学习参数(或者叫权重）的神经网络
2. 在输入数据集上迭代
3. 通过网络处理输入
4. 计算 loss (输出和正确答案的距离）
5. 将梯度反向传播给网络的参数
6. 更新网络的权重，一般使用一个简单的规则：`weight = weight - learning_rate * gradient`

Eg：AlexNet：

```python
class AlexNet(nn.Module):
    def __init__(self):
        super(AlexNet, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(1, 96, 11, 4), # in_channels, out_channels, kernel_size, stride, padding
            nn.ReLU(),
            nn.MaxPool2d(3, 2), # kernel_size, stride
            # 减小卷积窗口，使用填充为2来使得输入与输出的高和宽一致，且增大输出通道数
            nn.Conv2d(96, 256, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(3, 2),
            # 连续3个卷积层，且使用更小的卷积窗口。除了最后的卷积层外，进一步增大了输出通道数。
            # 前两个卷积层后不使用池化层来减小输入的高和宽
            nn.Conv2d(256, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 384, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(384, 256, 3, 1, 1),
            nn.ReLU(),
            nn.MaxPool2d(3, 2)
        )
         # 这里全连接层的输出个数比LeNet中的大数倍。使用丢弃层来缓解过拟合
        self.fc = nn.Sequential(
            nn.Linear(256*5*5, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(0.5),
            # 输出层。由于这里使用Fashion-MNIST，所以用类别数为10，而非论文中的1000
            nn.Linear(4096, 10),
        )

    def forward(self, img):
        feature = self.conv(img)
        output = self.fc(feature.view(img.shape[0], -1))
        return output
```

tips：一个模型的可学习参数可以通过`net.parameters()`返回

```python
params = list(net.parameters())
print(len(params))
print(params[0].size())  # conv1的权重
```

- `nn.Parameter `- 张量的一种，当它作为一个属性分配给一个`Module`时，它会被自动注册为一个参数。
- `autograd.Function` - 实现了自动求导前向和反向传播的定义，每个`Tensor`至少创建一个`Function`节点，该节点连接到创建`Tensor`的函数并对其历史进行编码。

### 3.5模型初始化

```python
# 查看随机初始化的conv参数
conv.weight.data
# 查看linear的参数
linear.weight.data
# 对conv进行kaiming初始化
torch.nn.init.kaiming_normal_(conv.weight.data)
conv.weight.data
# 对linear进行常数初始化
torch.nn.init.constant_(linear.weight.data,0.3)
linear.weight.data

#根据不同类型层，设定不同的权值初始化方法
def initialize_weights(self):
	for m in self.modules():
		# 判断是否属于Conv2d
		if isinstance(m, nn.Conv2d):
			torch.nn.init.xavier_normal_(m.weight.data)
			# 判断是否有偏置
			if m.bias is not None:
				torch.nn.init.constant_(m.bias.data,0.3)
		elif isinstance(m, nn.Linear):
			torch.nn.init.normal_(m.weight.data, 0.1)
			if m.bias is not None:
				torch.nn.init.zeros_(m.bias.data)
		elif isinstance(m, nn.BatchNorm2d):
			m.weight.data.fill_(1) 		 
			m.bias.data.zeros_()	
```

### 3.6损失函数

|        二分类交叉熵损失函数 torch.nn.BCELoss        |        计算二分类任务时的交叉熵（Cross Entropy）函数         |
| :-------------------------------------------------: | :----------------------------------------------------------: |
|       交叉熵损失函数torch.nn.CrossEntropyLoss       |                        计算交叉熵函数                        |
|              L1损失函数torch.nn.L1Loss              |       计算输出`y`和真实标签`target`之间的差值的绝对值        |
|             MSE损失函数torch.nn.MSELoss             |          计算输出`y`和真实标签`target`之差的平方。           |
|         平滑L1损失函数torch.nn.SmoothL1Loss         |          L1的平滑输出，其功能是减轻离群点带来的影响          |
| 目标泊松分布的负对数似然损失torch.nn.PoissonNLLLoss |                 泊松分布的负对数似然损失函数                 |
|              KL散度torch.nn.KLDivLoss               | 用于连续分布的距离度量，并且对离散采用的连续输出空间分布进行回归通常很有用 |
|             torch.nn.MarginRankingLoss              | 计算两个向量之间的相似度，用于排序任务。该方法用于计算两组数据之间的差异 |
|   多标签边界损失函数torch.nn.MultiLabelMarginLoss   |                对于多标签分类问题计算损失函数                |
|        二分类损失函数torch.nn.SoftMarginLoss        |                  计算二分类的 logistic 损失                  |
|      多分类的折页损失torch.nn.MultiMarginLoss       |                     计算多分类的折页损失                     |
|        三元组损失torch.nn.TripletMarginLoss         | 三元组：这是一种数据的存储或者使用格式。<实体1，关系，实体2>。在项目中，也可以表示为< `anchor`, `positive examples` , `negative examples`>。在这个损失函数中，我们希望去`anchor`的距离更接近`positive examples`，而远离`negative examples` |
|             torch.nn.HingeEmbeddingLoss             |             对输出的embedding结果做Hing损失计算              |
|       余弦相似度torch.nn.CosineEmbeddingLoss        | 对两个向量做余弦相似度。将余弦相似度作为一个距离的计算方式，如果两个向量的距离近，则损失函数值小，反之亦然。 |
|                  torch.nn.CTCLoss                   | 用于解决时序类数据的分类：计算连续时间序列和目标序列之间的损失。CTCLoss对输入和目标的可能排列的概率进行求和，产生一个损失值，这个损失值对每个输入节点来说是可分的 |

### 3.7训练和评估

首先应该设置模型的状态：如果是训练状态，那么模型的参数应该支持反向传播的修改；如果是验证/测试状态，则不应该修改模型参数。

如下的两个操作二选一即可：

```python
model.train()   # 训练状态
model.eval()   # 验证/测试状态
```

```python
#读取DataLoader中的全部数据
for data, label in train_loader:
#之后将数据放到GPU上用于后续计算，此处以.cuda()为例
data, label = data.cuda(), label.cuda()
#开始用当前批次数据做训练时，应当先将优化器的梯度置零：
optimizer.zero_grad()
#之后将data送入模型中训练
output = model(data)
#根据预先定义的criterion计算损失函数
loss = criterion(output, label)
#将loss反向传播回网络
loss.backward()
#使用优化器更新模型参数
optimizer.step()
```

验证/测试的流程基本与训练过程一致，不同点在于：

- 需要预先设置torch.no_grad，以及将model调至eval模式
- 不需要将优化器的梯度置零
- 不需要将loss反向回传到网络
- 不需要更新optimizer

### 3.9优化器

优化器是根据网络反向传播的梯度信息来更新网络的参数，以起到降低loss函数计算值，使得模型输出更加接近真实标签

`Optimizer`有三个属性：

- `defaults`：存储的是优化器的超参数
- `state`：参数的缓存
- `param_groups`：管理的参数组，是一个list，其中每个元素是一个字典，顺序是params，lr，momentum，dampening，weight_decay，nesterov

`Optimizer`有三个属性：

- `zero_grad()`：清空所管理参数的梯度，PyTorch的特性是张量的梯度不自动清零，因此每次反向传播后都需要清空梯度。

- `step()`：执行一步梯度更新，参数更新

- `add_param_group()`：添加参数组

- `load_state_dict()` ：加载状态参数字典，可以用来进行模型的断点续训练，继续上次的参数进行训练

- `state_dict()`：获取优化器当前状态信息字典

  #### 实际操作

```python
# 设置权重，服从正态分布  --> 2 x 2
weight = torch.randn((2, 2), requires_grad=True)
# 设置梯度为全1矩阵  --> 2 x 2
weight.grad = torch.ones((2, 2))
# 输出现有的weight和data
print("The data of weight before step:\n{}".format(weight.data))
print("The grad of weight before step:\n{}".format(weight.grad))
# 实例化优化器
optimizer = torch.optim.SGD([weight], lr=0.1, momentum=0.9)
# 进行一步操作
optimizer.step()
# 查看进行一步后的值，梯度
print("The data of weight after step:\n{}".format(weight.data))
print("The grad of weight after step:\n{}".format(weight.grad))
# 权重清零
optimizer.zero_grad()
# 检验权重是否为0
print("The grad of weight after optimizer.zero_grad():\n{}".format(weight.grad))
# 输出参数
print("optimizer.params_group is \n{}".format(optimizer.param_groups))
# 查看参数位置，optimizer和weight的位置一样，我觉得这里可以参考Python是基于值管理
print("weight in optimizer:{}\nweight in weight:{}\n".format(id(optimizer.param_groups[0]['params'][0]), id(weight)))
# 添加参数：weight2
weight2 = torch.randn((3, 3), requires_grad=True)
optimizer.add_param_group({"params": weight2, 'lr': 0.0001, 'nesterov': True})
# 查看现有的参数
print("optimizer.param_groups is\n{}".format(optimizer.param_groups))
# 查看当前状态信息
opt_state_dict = optimizer.state_dict()
print("state_dict before step:\n", opt_state_dict)
# 进行5次step操作
for _ in range(50):
    optimizer.step()
# 输出现有状态信息
print("state_dict after step:\n", optimizer.state_dict())
# 保存参数信息
torch.save(optimizer.state_dict(),os.path.join(r"D:\pythonProject\Attention_Unet", "optimizer_state_dict.pkl"))
print("----------done-----------")
# 加载参数信息
state_dict = torch.load(r"D:\pythonProject\Attention_Unet\optimizer_state_dict.pkl") # 需要修改为你自己的路径
optimizer.load_state_dict(state_dict)
print("load state_dict successfully\n{}".format(state_dict))
# 输出最后属性信息
print("\n{}".format(optimizer.defaults))
print("\n{}".format(optimizer.state))
print("\n{}".format(optimizer.param_groups))
```

**注意：**

1.每个优化器都是一个类，我们一定要进行实例化才能使用

2.记得梯度置零以及梯度更新

```python
optimizer = torch.optim.SGD(net.parameters(), lr=1e-5)
for epoch in range(EPOCH):
	...
	optimizer.zero_grad()  #梯度置零
	loss = ...             #计算loss
	loss.backward()        #BP反向传播
	optimizer.step()       #梯度更新
```

3.给网络不同的层赋予不同的优化器参数

```python
from torch import optim
from torchvision.models import resnet18
net = resnet18()

optimizer = optim.SGD([
    {'params':net.fc.parameters()},#fc的lr使用默认的1e-5
    {'params':net.layer4[0].conv1.parameters(),'lr':1e-2}],lr=1e-5)

# 可以使用param_groups查看属性
```

# 第四章、实战



# 第五章、模型定义



## 5.1模型定义

Sequential适用于快速验证结果，因为已经明确了要用哪些层，直接写一下就好了，不需要同时写__init__和forward；

ModuleList和ModuleDict在某个完全相同的层需要重复出现多次时，非常方便实现，可以”一行顶多行“；

当我们需要之前层的信息的时候，比如 ResNets 中的残差计算，当前层的结果需要和之前层中的结果进行融合，一般使用 ModuleList/ModuleDict 比较方便。

### 5.1.1Sequential

适用于简单串联各个层，可以通过两种方式排列

```python
net = nn.Sequential(
        nn.Linear(784, 256),
        nn.ReLU(),
        nn.Linear(256, 10), 
        )
```

```
net2 = nn.Sequential(collections.OrderedDict([
          ('fc1', nn.Linear(784, 256)),
          ('relu1', nn.ReLU()),
          ('fc2', nn.Linear(256, 10))
          ]))
print(net2)
```

### 5.1.2ModuleList

ModuleList 接收一个子模块（或层，需属于nn.Module类）的列表作为输入，然后也可以类似List那样进行append和extend操作。同时，子模块或层的权重也会自动添加到网络中来。

它是一个储存不同 module，并自动将每个 module 的 parameters 添加到网络之中的容器。

```python
net = nn.ModuleList([nn.Linear(784, 256), nn.ReLU()])
net.append(nn.Linear(256, 10)) # # 类似List的append操作
print(net[-1])  # 类似List的索引访问,输出256的那个
print(net)#784,256
```

要特别注意的是，nn.ModuleList 并没有定义一个网络，它只是将不同的模块储存在一起。ModuleList中元素的先后顺序并不代表其在网络中的真实位置顺序，需要经过forward函数指定各个层的先后顺序后才算完成了模型的定义。具体实现时用for循环即可完成。网络的执行顺序是根据 forward 函数来决定的

```python
class model(nn.Module):
  def __init__(self, ...):
    super().__init__()
    self.modulelist = ...
    ...
    
  def forward(self, x):
    for layer in self.modulelist:
      x = layer(x)
    return x
```

```python
#包含两个全连接层
class net1(nn.Module):
    def __init__(self):
        super(net1, self).__init__()
        self.linears = nn.ModuleList([nn.Linear(10,10) for i in range(2)])
    def forward(self, x):
        for m in self.linears:
            x = m(x)
        return x
```



### 5.1.3ModuleDict

ModuleDict能够更方便地为神经网络的层添加名称

```python
net = nn.ModuleDict({
    'linear': nn.Linear(784, 256),
    'act': nn.ReLU(),
})
net['output'] = nn.Linear(256, 10) # 添加
print(net['linear']) # 访问
print(net.output)
print(net)
```

## 5.2利用模型块快速搭建复杂网络

虽然模型有很多层， 但是其中有很多重复出现的结构。考虑到每一层有其输入和输出，若干层串联成的”模块“也有其输入和输出，如果我们能将这些重复出现的层定义为一个”模块“，每次只需要向网络中添加对应的模块来构建模型，这样将会极大便利模型构建的过程。

https://github.com/datawhalechina/thorough-pytorch/blob/main/source/%E7%AC%AC%E4%BA%94%E7%AB%A0/5.2%20%E5%88%A9%E7%94%A8%E6%A8%A1%E5%9E%8B%E5%9D%97%E5%BF%AB%E9%80%9F%E6%90%AD%E5%BB%BA%E5%A4%8D%E6%9D%82%E7%BD%91%E7%BB%9C.md

## 5.3修改模型

### 5.3.1修改模型层

```python
#以rennet50（）为例
import torchvision.models as models
net = models.resnet50()
#修改其中的fc部分
from collections import OrderedDict
classifier = nn.Sequential(OrderedDict([('fc1', nn.Linear(2048, 128)),
                          ('relu1', nn.ReLU()), 
                          ('dropout1',nn.Dropout(0.5)),
                          ('fc2', nn.Linear(128, 10)),
                          ('output', nn.Softmax(dim=1))
                          ]))
    
net.fc = classifier#这里的操作相当于将模型（net）最后名称为“fc”的层替换成了名称为“classifier”的结构
```

### 5.3.2添加外部输入

基本思路是：将原模型添加输入位置前的部分作为一个整体，同时在forward中定义好原模型不变的部分、添加的输入和后续层之间的连接关系，从而完成模型的修改

```python
#我们希望利用已有的模型结构，在倒数第二层增加一个额外的输入变量add_variable来辅助预测
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc_add = nn.Linear(1001, 10, bias=True)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x, add_variable):
        x = self.net(x)
        x = torch.cat((self.dropout(self.relu(x)), add_variable.unsqueeze(1)),1)
        x = self.fc_add(x)
        x = self.output(x)
        return x
```

**要点**：

1.通过torch.cat实现了tensor的拼接

2.resnet50输出是一个1000维的tensor，我们通过修改forward函数（配套定义一些层），先将2048维的tensor通过激活函数层和dropout层，再和外部输入变量"add_variable"拼接，最后通过全连接层映射到指定的输出维度10。

3.进行unsqueeze操作是为了和net输出的tensor保持维度一致，常用于add_variable是单一数值 (scalar) 的情况，此时add_variable的维度是 (batch_size, )，需要在第二维补充维数1，从而可以和tensor进行torch.cat操作。

```
#使用：
import torchvision.models as models
net = models.resnet50()
model = Model(net).cuda()
```

### 5.3.3添加额外输出

基本的思路是修改模型定义中forward函数的return变量。

```python
class Model(nn.Module):
    def __init__(self, net):
        super(Model, self).__init__()
        self.net = net
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
        self.fc1 = nn.Linear(1000, 10, bias=True)
        self.output = nn.Softmax(dim=1)
        
    def forward(self, x, add_variable):
        x1000 = self.net(x)
        x10 = self.dropout(self.relu(x1000))
        x10 = self.fc1(x10)
        x10 = self.output(x10)
        return x10, x1000
```

使用：

```python
import torchvision.models as models
net = models.resnet50()
model = Model(net).cuda()
out10, out1000 = model(inputs, add_var)
```

## 5.4模型的保存与读取

格式：pkl，pt，pth

内容：主要包含两个部分：模型结构和权重。其中模型是继承nn.Module的类，权重的数据结构是一个字典（key是层名，value是权重向量）。存储也由此分为两种形式：存储整个模型（包括结构和权重），和只存储模型权重。

单卡与多卡的存储区别：PyTorch中将模型和数据放到GPU上有两种方式——.cuda()和.to(device)，本节后续内容针对前一种方式进行讨论。如果要使用多卡训练的话，需要对模型使用torch.nn.DataParallel

仅记录单卡保存+单卡加载，其余的暂时用不到？

```python
import os
import torch
from torchvision import models
#在使用os.envision命令指定使用的GPU后，即可进行模型保存和读取操作。注意这里即便保存和读取时使用的GPU不同也无妨
os.environ['CUDA_VISIBLE_DEVICES'] = '0'   #这里替换成希望使用的GPU编号
model = models.resnet152(pretrained=True)
model.cuda()

# 保存+读取整个模型
torch.save(model, save_dir)
loaded_model = torch.load(save_dir)
loaded_model.cuda()

# 保存+读取模型权重
torch.save(model.state_dict(), save_dir)
loaded_dict = torch.load(save_dir)
loaded_model = models.resnet152()   #注意这里需要对模型结构有定义
loaded_model.state_dict = loaded_dict
loaded_model.cuda()
```

# 第六章

## 6.1自定义损失函数

```python
#以函数方式定义
def my_loss(output, target):
    loss = torch.mean((output - target)**2)
    return loss
```

如果看每一个损失函数的继承关系我们就可以发现`Loss`函数部分继承自`_loss`, 部分继承自`_WeightedLoss`, 而`_WeightedLoss`继承自`_loss`，` _loss`继承自 **nn.Module**。我们可以将其当作神经网络的一层来对待，同样地，我们的损失函数类就需要继承自**nn.Module**类，

```python
class DiceLoss(nn.Module):
    def __init__(self,weight=None,size_average=True):
        super(DiceLoss,self).__init__()
        
	def forward(self,inputs,targets,smooth=1):
        inputs = F.sigmoid(inputs)       
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        intersection = (inputs * targets).sum()                   
        dice = (2.*intersection + smooth)/(inputs.sum() + targets.sum() + smooth)  
        return 1 - dice

# 使用方法    
criterion = DiceLoss()
loss = criterion(input,targets)
```

## 6.2动态调整学习率

学习速率设置过小，会极大降低收敛速度，增加训练时间；学习率太大，可能导致参数在最优解两侧来回振荡

设置方式：scheduler

封装好的动态调整学习率的方法：

- [`lr_scheduler.LambdaLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.LambdaLR.html#torch.optim.lr_scheduler.LambdaLR)
- [`lr_scheduler.MultiplicativeLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiplicativeLR.html#torch.optim.lr_scheduler.MultiplicativeLR)
- [`lr_scheduler.StepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.StepLR.html#torch.optim.lr_scheduler.StepLR)
- [`lr_scheduler.MultiStepLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.MultiStepLR.html#torch.optim.lr_scheduler.MultiStepLR)
- [`lr_scheduler.ExponentialLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ExponentialLR.html#torch.optim.lr_scheduler.ExponentialLR)
- [`lr_scheduler.CosineAnnealingLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingLR.html#torch.optim.lr_scheduler.CosineAnnealingLR)
- [`lr_scheduler.ReduceLROnPlateau`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.ReduceLROnPlateau.html#torch.optim.lr_scheduler.ReduceLROnPlateau)
- [`lr_scheduler.CyclicLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CyclicLR.html#torch.optim.lr_scheduler.CyclicLR)
- [`lr_scheduler.OneCycleLR`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html#torch.optim.lr_scheduler.OneCycleLR)
- [`lr_scheduler.CosineAnnealingWarmRestarts`](https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.CosineAnnealingWarmRestarts.html#torch.optim.lr_scheduler.CosineAnnealingWarmRestarts)

```python
#使用
# 选择一种优化器
optimizer = torch.optim.Adam(...) 
# 选择上面提到的一种或多种动态调整学习率的方法
scheduler1 = torch.optim.lr_scheduler.... 
scheduler2 = torch.optim.lr_scheduler....
...
schedulern = torch.optim.lr_scheduler....
# 进行训练
for epoch in range(100):
    train(...)
    validate(...)
    optimizer.step()
    # 需要在优化器参数更新之后再动态调整学习率
	scheduler1.step() 
	...
    schedulern.step()
    #我们在使用官方给出的torch.optim.lr_scheduler时，需要将scheduler.step()放在optimizer.step()后面进行使用。
```

## 6.3自定义scheduler

方案：自定义函数`adjust_learning_rate`来改变`param_group`中`lr`的值

```python
#需要学习率每30轮下降为原来的1/10
def adjust_learning_rate(optimizer, epoch):
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
```

使用：

```python
def adjust_learning_rate(optimizer,...):
    ...
optimizer = torch.optim.SGD(model.parameters(),lr = args.lr,momentum = 0.9)
for epoch in range(10):
    train(...)
    validate(...)
    adjust_learning_rate(optimizer,epoch)
```

## 6.4半精度训练

半精度：torch.float16，数位减了一半

```python
from torch.cuda.amp import autocast
```

在模型定义中，使用python的装饰器方法，用autocast装饰模型中的forward函数。

关于装饰器的使用，可以参考[这里](https://www.cnblogs.com/jfdwd/p/11253925.html)：

```python
@autocast()   
def forward(self, x):
    ...
    return x
```

训练过程：只需在将数据输入模型及其之后的部分放入“with autocast():“即可：

```python
 for x in train_loader:
	x = x.cuda()
	with autocast():
        output = model(x)
        ...
```

## 6.5数据增强imgaug

不搞cv方向，不学。

有空：https://github.com/datawhalechina/thorough-pytorch/blob/main/source/%E7%AC%AC%E5%85%AD%E7%AB%A0/6.5%20%E6%95%B0%E6%8D%AE%E5%A2%9E%E5%BC%BA-imgaug.md

## 6.6使用argparse进行调参

作用：可以直接在命令行中就可以向程序中传入参数。

使用：分为三个步骤：

- 创建`ArgumentParser()`对象
- 调用`add_argument()`方法添加参数
- 使用`parse_args()`解析参数 在接下来的内容中，我们将以实际操作来学习argparse的使用方法。

未完成

## 6.7技巧

未完成

# 第七章、网络结构可视化

## 7.1可视化网络结构

使用**torchinfo**进行结构化输出，只需要使用torchinfo.summary()就行

```
import torchvision.models as models
from torchinfo import summary
resnet18 = models.resnet18() # 实例化模型
summary(resnet18, (1, 3, 224, 224)) # 1：batch_size 3:图片的通道数 224: 图片的高宽
```

## 7.2卷积可视化

### 7.2.1卷积核

```python
conv1 = dict(model.features.named_children())['3']
#以第“3”层为例，可视化对应的参数
kernel_set = conv1.weight.detach()
num = len(conv1.weight.detach())
print(kernel_set.shape)
for i in range(0,num):
    i_kernel = kernel_set[i]
    plt.figure(figsize=(20, 17))
    if (len(i_kernel)) > 1:
        for idx, filer in enumerate(i_kernel):
            plt.subplot(9, 9, idx+1) 
            plt.axis('off')
            plt.imshow(filer[ :, :].detach(),cmap='bwr')
```

### 7.2.2CNN特征图可视化方法

特征图：输入的原始图像经过每次卷积层得到的数据，可视化卷积核是为了看模型提取哪些特征，可视化特征图则是为了看模型提取到的特征是什么样子的

```python
class Hook(object):
    def __init__(self):
        self.module_name = []
        self.features_in_hook = []
        self.features_out_hook = []

    def __call__(self,module, fea_in, fea_out):
        print("hooker working", self)
        self.module_name.append(module.__class__)
        self.features_in_hook.append(fea_in)
        self.features_out_hook.append(fea_out)
        return None
    

def plot_feature(model, idx, inputs):
    hh = Hook()
    model.features[idx].register_forward_hook(hh)
    
    # forward_model(model,False)
    model.eval()
    _ = model(inputs)
    print(hh.module_name)
    print((hh.features_in_hook[0][0].shape))
    print((hh.features_out_hook[0].shape))
    
    out1 = hh.features_out_hook[0]

    total_ft  = out1.shape[1]
    first_item = out1[0].cpu().clone()    

    plt.figure(figsize=(20, 17))
    

    for ftidx in range(total_ft):
        if ftidx > 99:
            break
        ft = first_item[ftidx]
        plt.subplot(10, 10, ftidx+1) 
        
        plt.axis('off')
        #plt.imshow(ft[ :, :].detach(),cmap='gray')
        plt.imshow(ft[ :, :].detach())
```

## 7.3FlashTorch实现CNN可视化

假如报错：[https://github.com/MisaOgura/flashtorch/issues/39](https://github.com/MisaOgura/flashtorch/issues/39）)

```python
#可视化梯度
import matplotlib.pyplot as plt
import torchvision.models as models
from flashtorch.utils import apply_transforms, load_image
from flashtorch.saliency import Backprop

model = models.alexnet(pretrained=True)
backprop = Backprop(model)

image = load_image('/content/images/great_grey_owl.jpg')
owl = apply_transforms(image)

target_class = 24
backprop.visualize(owl, target_class, guided=True, use_gpu=True)
```

```python
#可视化卷积核
import torchvision.models as models
from flashtorch.activmax import GradientAscent

model = models.vgg16(pretrained=True)
g_ascent = GradientAscent(model.features)

# specify layer and filter info
conv5_1 = model.features[24]
conv5_1_filters = [45, 271, 363, 489]

g_ascent.visualize(conv5_1, conv5_1_filters, title="VGG16: conv5_1")
```

## 7.4TensorBoard

