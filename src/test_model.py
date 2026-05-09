import numpy as np
import os
import matplotlib.pyplot as plt
from activations import ReLU, Sigmoid, Tanh  # 假设你已有的文件
import dataset
import model


def cross_entropy_loss(logits, y_true, model, weight_decay=1e-4):
    m = y_true.shape[0]
    # Softmax 计算 softmax
    exps = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = exps / np.sum(exps, axis=1, keepdims=True)
    
    # 交叉熵
    log_likelihood = -np.log(probs[range(m), y_true] + 1e-15)
    data_loss = np.sum(log_likelihood) / m
    
    # L2 正则化损耗
    l2_loss = 0
    for w in model.weights.values():
        l2_loss += 0.5 * weight_decay * np.sum(w * w)
    
    total_loss = data_loss + l2_loss
    
    # 反向传播的起点梯度 利用链式法则进行反向传播dL/dZ
    dx = probs.copy()
    dx[range(m), y_true] -= 1
    dx /= m
    
    return total_loss, dx

def get_confusion_matrix(y_true, y_pred, num_classes=10):
    cm = np.zeros((num_classes, num_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        cm[t, p] += 1
    return cm

# ==========================================
# 2. 优化器 (Optimizer with LR Decay)
# ==========================================
class SGDOptimizer:
    def __init__(self, params, lr=0.1, lr_decay=0.95, weight_decay=1e-4):
        self.params = params
        self.lr = lr
        self.lr_decay = lr_decay
        self.weight_decay = weight_decay

    def step(self, grads):
        for key in self.params:
            # 应用 L2 梯度并在更新前更新权重 (W = W - lr * (grad + wd * W))
            grad = grads[key]
            if key.startswith('W'):
                grad += self.weight_decay * self.params[key]
            self.params[key] -= self.lr * grad

    def decay_lr(self):
        self.lr *= self.lr_decay

# ==========================================
# 3. 核心训练函数 (Satisfies Requirements)
# ==========================================
def train_model(model, data, config):
    x_train, x_val, y_train, y_val = data
    optimizer = SGDOptimizer(
        model.get_parameters(), 
        lr=config['lr'], 
        lr_decay=config['lr_decay'],
        weight_decay=config['weight_decay']
    )
    
    history = {'train_loss': [], 'val_acc': []}
    best_val_acc = 0.0
    
    print(f"Starting Training: {model}")
    
    for epoch in range(config['epochs']):
        epoch_loss = 0
        # 训练循环
        for x_batch, y_batch in dataset.get_batches(x_train, y_train, config['batch_size']):
            # 前向
            logits = model.forward(x_batch, training=True)
            # 计算 Loss & 梯度起点
            loss, dL_dout = cross_entropy_loss(logits, y_batch, model, config['weight_decay'])
            # 反向
            model.backward(dL_dout)
            # 更新
            optimizer.step(model.get_gradients())
            epoch_loss += loss
            
        # 验证集评估
        val_logits = model.forward(x_val, training=False)
        val_preds = np.argmax(val_logits, axis=1)
        val_acc = np.mean(val_preds == y_val)
        
        # 记录历史
        history['train_loss'].append(epoch_loss / (len(x_train)//config['batch_size']))
        history['val_acc'].append(val_acc)
        
        # 学习率衰减
        optimizer.decay_lr()
        
        # 自动保存最优模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model.save_weights("best_model.npz")
            print(f"Epoch {epoch+1}: New best Val Acc: {val_acc:.4f} (Saved)")
        else:
            print(f"Epoch {epoch+1}: Val Acc: {val_acc:.4f}")
            
    return history

# ==========================================
# 4. 提交要求：可视化与分析 (Post-process)
# ==========================================
def visualize_results(history, model, x_test, y_test):
    # (1) 绘制 Loss 和 Acc 曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history['train_loss'], label='Train Loss')
    plt.title('Loss Curve')
    plt.subplot(1, 2, 2)
    plt.plot(history['val_acc'], label='Val Acc')
    plt.title('Validation Accuracy')
    plt.show()

    # (2) 权重可视化 (权重矩阵恢复成图像尺寸)
    # 假设第一层 W1 形状是 (784, hidden)
    W1 = model.weights['W1']
    plt.figure(figsize=(10, 10))
    for i in range(min(16, W1.shape[1])):
        plt.subplot(4, 4, i+1)
        weight_img = W1[:, i].reshape(28, 28)
        plt.imshow(weight_img, cmap='RdBu') # 红蓝代表正负权重
        plt.axis('off')
    plt.suptitle("First Layer Weights Visualization")
    plt.show()

    # (3) 混淆矩阵与错例分析
    test_logits = model.forward(x_test, training=False)
    test_preds = np.argmax(test_logits, axis=1)
    
    cm = get_confusion_matrix(y_test, test_preds)
    print("\nConfusion Matrix:")
    print(cm)
    
    # 寻找错例
    errors = np.where(test_preds != y_test)[0]
    plt.figure(figsize=(10, 4))
    for i in range(min(5, len(errors))):
        idx = errors[i]
        plt.subplot(1, 5, i+1)
        plt.imshow(x_test[idx].reshape(28, 28), cmap='gray')
        plt.title(f"True:{y_test[idx]}\nPred:{test_preds[idx]}")
        plt.axis('off')
    plt.show()

# ==========================================
# 5. 主运行程序
# ==========================================
if __name__ == "__main__":
    # 加载数据 (使用你之前写的 load_fashion_mnist)
    x_train, x_val, x_test, y_train, y_val, y_test = dataset.load_fashion_mnist()
    
    # 超参数配置 (你可以手动改这里进行“参数查找”)
    config = {
        'lr': 0.1,
        'lr_decay': 0.96,
        'weight_decay': 1e-4,
        'epochs': 20,
        'batch_size': 128,
        'hidden_dims': [256, 128]
    }
    
    # 初始化
    mlp = model.MLP(input_dim=784, hidden_dims=config['hidden_dims'], output_dim=10, activation='relu')
    
    # 训练
    history = train_model(mlp, (x_train, x_val, y_train, y_val), config)
    
    # 加载最优模型进行最终测试
    mlp.load_weights("best_model.npz")
    visualize_results(history, mlp, x_test, y_test)