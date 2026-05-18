# computer-vision-hw1-
# Model.py 设计说明

##  核心功能

`model.py` 实现了一个完整的三层神经网络（MLP），包含：
-  完整的前向传播和反向传播
-  支持自定义隐藏层大小
- 支持三种激活函数（ReLU, Sigmoid, Tanh）
  - 自动微分（手工实现反向传播）
-  权重保存和加载
-  L2正则化支持

##  网络架构

```
Input (784) -> Hidden1 (256) -> Hidden2 (128) -> Output (10)
               ↓ ReLU          ↓ ReLU            ↓ Logits
```

**三层含义**：输入层 + 隐藏层（可多个）+ 输出层

##  关键设计点

### 1. 参数初始化

```python
# He初始化（适用于ReLU）
std = np.sqrt(2.0 / input_size)

# Xavier初始化（适用于Sigmoid/Tanh）
std = np.sqrt(1.0 / input_size)
```

**为什么这样初始化？**
- 防止梯度消失/爆炸
- ReLU会"杀死"一半的神经元，需要更大的初始化方差
- Xavier适用于对称激活函数（Sigmoid, Tanh）

### 2. 前向传播

```python
def forward(self, X, training=True):
    # 每一层：
    # 1. 线性变换: Z = A * W + b
    # 2. 激活函数: A = activation(Z)
    # 3. 保存中间变量（用于反向传播）
```

**重要细节**：
- 输出层**不使用激活函数**，直接输出logits
- 这样设计是为了配合交叉熵损失函数（数值稳定性更好）
- `training=True` 时保存中间变量用于反向传播

### 3. 反向传播（自动微分的核心）

使用**链式法则**计算梯度：

```
dL/dW = dL/dA * dA/dZ * dZ/dW
```

**实现细节**：
```python
# 1. 激活函数的导数
if layer_idx < self.num_layers:
    dZ = dA * self.activation.backward(Z)
else:
    dZ = dA  # 输出层

# 2. 权重梯度
gradients['W'] = A_prev.T @ dZ / batch_size

# 3. 偏置梯度
gradients['b'] = sum(dZ) / batch_size

# 4. 传递到前一层
dA = dZ @ W.T
```

**为什么除以batch_size？**
- 使用mini-batch SGD时，梯度是所有样本梯度的平均值
- 这样学习率的设置不依赖于batch size

### 4. L2正则化

```python
# 在梯度中添加L2惩罚项
gradients['W'] += weight_decay * weights['W']
```

**等价于损失函数**：
```
L = CrossEntropy + (weight_decay / 2) * ||W||²
```

### 5. 预测函数

```python
# 预测类别
def predict(self, X):
    logits = self.forward(X, training=False)
    return np.argmax(logits, axis=1)

# 预测概率
def predict_proba(self, X):
    logits = self.forward(X, training=False)
    # 手动实现softmax（数值稳定版本）
    exp_logits = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    return exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
```

**数值稳定性技巧**：
- Softmax计算时减去最大值，防止exp溢出

##  使用示例

### 基本使用

```python
from model import MLP

# 创建模型
model = MLP(
    input_dim=784,
    hidden_dims=[256, 128],  # 两个隐藏层
    output_dim=10,
    activation='relu'
)

# 前向传播
X = np.random.randn(32, 784)  # batch_size=32
logits = model.forward(X, training=True)

# 反向传播（需要先计算loss的梯度）
dL_dout = compute_loss_gradient(logits, y)
gradients = model.backward(dL_dout)

# 添加L2正则化
model.add_l2_gradients(weight_decay=0.01)

# 预测
predictions = model.predict(X_test)
probabilities = model.predict_proba(X_test)
```

### 保存和加载

```python
# 保存
model.save_weights('weights/best_model.npz')

# 加载
model.load_weights('weights/best_model.npz')
```

##  与其他模块的配合

### 与 loss.py 配合

```python
# loss.py 应该实现：
loss, dL_dout = cross_entropy_loss(logits, y)
model.backward(dL_dout)
```

### 与 optimizer.py 配合

```python
# optimizer.py 应该实现：
optimizer = SGD(learning_rate=0.01)
params = model.get_parameters()
gradients = model.get_gradients()
updated_params = optimizer.step(params, gradients)
model.set_parameters(updated_params)
```

### 与 trainer.py 配合

```python
# 训练循环
for epoch in range(num_epochs):
    logits = model.forward(X_train, training=True)
    loss, dL_dout = loss_fn(logits, y_train)
    gradients = model.backward(dL_dout)
    model.add_l2_gradients(weight_decay)
    optimizer.step(model, gradients)
```

##  注意事项

1. **输出层不使用激活函数**
   - 输出原始logits，配合交叉熵损失
   - 不要在输出后手动加softmax

2. **梯度计算前必须先forward**
   - backward依赖forward保存的中间变量
   - 确保training=True

3. **L2正则化只应用于权重，不应用于偏置**
   - `add_l2_gradients`只修改权重的梯度

4. **数值稳定性**
   - Sigmoid/Tanh使用clip防止溢出
   - Softmax减去最大值防止exp溢出

##  测试验证

运行 `test_model.py` 验证：
-  前向传播输出shape正确
-  反向传播梯度shape正确
-  概率和为1（softmax正确）
-  保存/加载功能正常
-  参数总数计算正确

##  性能优化建议

1. **使用向量化操作**（已实现）
   - 全部使用NumPy矩阵运算
   - 避免Python循环

2. **批量处理**（已支持）
   - forward/backward都支持batch输入
   - 梯度自动平均

3. **内存效率**
   - training=False时不保存中间变量
   - 减少推理时的内存占用
