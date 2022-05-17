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

###3.1步骤
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