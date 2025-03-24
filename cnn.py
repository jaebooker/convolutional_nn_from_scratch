import torch
import time
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Subset
import matplotlib.pyplot as plt
import numpy as np

torch.manual_seed(32)

transform_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

transform_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

train_dataset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
test_dataset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_test)

train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def batch_norm(x, eps=1e-5):
    """Batch normalization"""
    batch_size = x.shape[0]
    mean = x.mean(dim=0, keepdim=True)
    var = x.var(dim=0, unbiased=False, keepdim=True)
    x_norm = (x - mean) / torch.sqrt(var + eps)
    return x_norm


def conv2d(input_tensor, kernel, stride=1, padding=0):
    batch_size, in_channels, in_height, in_width = input_tensor.shape
    out_channels, _, kernel_height, kernel_width = kernel.shape
    if padding > 0:
        padding_layer = torch.zeros((batch_size, in_channels, 
                                   in_height + 2*padding, 
                                   in_width + 2*padding))
        padding_layer[:, :, padding:-padding, padding:-padding] = input_tensor
        input_tensor = padding_layer
        in_height += 2*padding
        in_width += 2*padding

    out_height = (in_height - kernel_height) // stride + 1
    out_width = (in_width - kernel_width) // stride + 1
    
    patches = torch.zeros(batch_size, out_height, out_width, 
                         in_channels, kernel_height, kernel_width)
    
    for h in range(0, out_height):
        h_start = h * stride
        for w in range(0, out_width):
            w_start = w * stride
            patches[:, h, w] = input_tensor[
                :, :,
                h_start:h_start + kernel_height,
                w_start:w_start + kernel_width
            ]
    
    patches = patches.reshape(batch_size, out_height * out_width, -1)
    reshaped_kernel = kernel.reshape(out_channels, -1)
    output = torch.matmul(patches, reshaped_kernel.T)
    output = output.reshape(batch_size, out_height, out_width, out_channels)
    output = output.permute(0, 3, 1, 2)
    
    return output


def max_pool2d(input_tensor, kernel_size=2, stride=None):

    if stride is None:
        stride = kernel_size
        
    batch_size, channels, height, width = input_tensor.shape
    out_height = (height - kernel_size) // stride + 1
    out_width = (width - kernel_size) // stride + 1
    output = input_tensor.unfold(2, kernel_size, stride).unfold(3, kernel_size, stride)
    
    output = output.contiguous().view(
        batch_size, channels, out_height, out_width, kernel_size * kernel_size
    )
    output = torch.max(output, dim=-1)[0]
    
    return output


def relu(x):
    return torch.maximum(x, torch.tensor(0.0))


def softmax(Z):
    exp_Z = torch.exp(Z - torch.max(Z, dim=0)[0])
    return exp_Z / torch.sum(exp_Z, dim=0)


def cross_entropy_loss(A, y):
    m = y.shape[0]
    y_one_hot = torch.zeros((m, A.shape[0]), dtype=torch.float32)
    y_one_hot[range(m), y] = 1
    y_one_hot = y_one_hot.T
    return -torch.mean(torch.sum(y_one_hot * torch.log(A + 1e-8), dim=0))


def init_weights(shape, method='he'):
    if method == 'he':
        fan_in = shape[1] if len(shape) == 2 else np.prod(shape[1:])
        std = np.sqrt(2.0 / fan_in)
        return torch.randn(*shape) * std
    return torch.randn(*shape) * 0.01


def network_with_conv(X, y, learning_rate, epochs):
    conv1_weights = torch.nn.Parameter(init_weights((16, 3, 3, 3)))
    conv2_weights = torch.nn.Parameter(init_weights((32, 16, 3, 3)))
    conv3_weights = torch.nn.Parameter(init_weights((64, 32, 3, 3)))
    
    n_after_conv = 4 * 4 * 64 
    W1 = torch.nn.Parameter(init_weights((512, n_after_conv)))
    b1 = torch.nn.Parameter(torch.zeros(512))
    
    W2 = torch.nn.Parameter(init_weights((256, 512)))
    b2 = torch.nn.Parameter(torch.zeros(256))
    
    W3 = torch.nn.Parameter(init_weights((128, 256)))
    b3 = torch.nn.Parameter(torch.zeros(128))
    
    W4 = torch.nn.Parameter(init_weights((64, 128)))
    b4 = torch.nn.Parameter(torch.zeros(64))
    
    W5 = torch.nn.Parameter(init_weights((10, 64)))
    b5 = torch.nn.Parameter(torch.zeros(10))

    v_conv1 = torch.zeros_like(conv1_weights)
    v_conv2 = torch.zeros_like(conv2_weights)
    v_conv3 = torch.zeros_like(conv3_weights)
    v_W1, v_b1 = torch.zeros_like(W1), torch.zeros_like(b1)
    v_W2, v_b2 = torch.zeros_like(W2), torch.zeros_like(b2)
    v_W3, v_b3 = torch.zeros_like(W3), torch.zeros_like(b3)
    v_W4, v_b4 = torch.zeros_like(W4), torch.zeros_like(b4)
    v_W5, v_b5 = torch.zeros_like(W5), torch.zeros_like(b5)
    
    beta = 0.9  
    cost_values = []
    test_acc_history = []
    train_acc_history = []  
    total_batches = len(train_loader)

    for epoch in range(epochs):
        epoch_loss = 0
        print(f"\nEpoch: {epoch}")
        
        train_correct, train_total = 0, 0
        
        for batch_idx, data in enumerate(train_loader):
            inputs, labels = data
            batch_size = inputs.shape[0]
            y = labels

            conv1_output = conv2d(inputs, conv1_weights, stride=1, padding=1)
            conv1_bn = batch_norm(conv1_output)
            conv1_activated = relu(conv1_bn)
            pooled1 = max_pool2d(conv1_activated, kernel_size=2)
            
            conv2_output = conv2d(pooled1, conv2_weights, stride=1, padding=1)
            conv2_bn = batch_norm(conv2_output)
            conv2_activated = relu(conv2_bn)
            pooled2 = max_pool2d(conv2_activated, kernel_size=2)
            
            conv3_output = conv2d(pooled2, conv3_weights, stride=1, padding=1)
            conv3_bn = batch_norm(conv3_output)
            conv3_activated = relu(conv3_bn)
            pooled3 = max_pool2d(conv3_activated, kernel_size=2)
            
            flattened = pooled3.reshape(batch_size, -1)
            
            Z1 = torch.mm(W1, flattened.T)
            Z1 = Z1 + b1.view(-1, 1)
            Z1_bn = batch_norm(Z1)
            A1 = relu(Z1_bn)
            
            Z2 = torch.mm(W2, A1)
            Z2 = Z2 + b2.view(-1, 1)
            Z2_bn = batch_norm(Z2)
            A2 = relu(Z2_bn)
            
            Z3 = torch.mm(W3, A2)
            Z3 = Z3 + b3.view(-1, 1)
            Z3_bn = batch_norm(Z3)
            A3 = relu(Z3_bn)
            
            Z4 = torch.mm(W4, A3)
            Z4 = Z4 + b4.view(-1, 1)
            Z4_bn = batch_norm(Z4)
            A4 = relu(Z4_bn)
            
            Z5 = torch.mm(W5, A4)
            Z5 = Z5 + b5.view(-1, 1)
            A5 = softmax(Z5)
            
            predicted = torch.argmax(A5, dim=0)
            train_correct += (predicted == y).sum().item()
            train_total += y.size(0)
            
            J = cross_entropy_loss(A5, y)
            epoch_loss += J.item()
            
            if (batch_idx + 1) % 100 == 0:
                print(f"  Batch {batch_idx + 1}/{total_batches}, Loss: {J.item():.4f}")
            
            J.backward()

            with torch.no_grad():
                
                v_conv1 = beta * v_conv1 + learning_rate * conv1_weights.grad
                conv1_weights -= v_conv1
                
                v_conv2 = beta * v_conv2 + learning_rate * conv2_weights.grad
                conv2_weights -= v_conv2
                
                v_conv3 = beta * v_conv3 + learning_rate * conv3_weights.grad
                conv3_weights -= v_conv3
                
                v_W1 = beta * v_W1 + learning_rate * W1.grad
                v_b1 = beta * v_b1 + learning_rate * b1.grad
                W1 -= v_W1
                b1 -= v_b1
                
                v_W2 = beta * v_W2 + learning_rate * W2.grad
                v_b2 = beta * v_b2 + learning_rate * b2.grad
                W2 -= v_W2
                b2 -= v_b2
                
                v_W3 = beta * v_W3 + learning_rate * W3.grad
                v_b3 = beta * v_b3 + learning_rate * b3.grad
                W3 -= v_W3
                b3 -= v_b3
                
                v_W4 = beta * v_W4 + learning_rate * W4.grad
                v_b4 = beta * v_b4 + learning_rate * b4.grad
                W4 -= v_W4
                b4 -= v_b4
                
                v_W5 = beta * v_W5 + learning_rate * W5.grad
                v_b5 = beta * v_b5 + learning_rate * b5.grad
                W5 -= v_W5
                b5 -= v_b5

                conv1_weights.grad.zero_()
                conv2_weights.grad.zero_()
                conv3_weights.grad.zero_()
                W1.grad.zero_()
                b1.grad.zero_()
                W2.grad.zero_()
                b2.grad.zero_()
                W3.grad.zero_()
                b3.grad.zero_()
                W4.grad.zero_()
                b4.grad.zero_()
                W5.grad.zero_()
                b5.grad.zero_()

            cost_values.append(J.item())
        
        train_acc = (train_correct / train_total) * 100
        train_acc_history.append(train_acc)
        
        print(f"  Epoch {epoch} average loss: {epoch_loss/total_batches:.4f}")
        print(f"  Training Accuracy: {train_acc:.2f}%")
        
        test_acc = evaluate_model(conv1_weights, conv2_weights, conv3_weights, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5)
        test_acc_history.append(test_acc)

    return cost_values, test_acc_history, train_acc_history


def evaluate_model(conv1_weights, conv2_weights, conv3_weights, W1, b1, W2, b2, W3, b3, W4, b4, W5, b5):
    correct, total = 0, 0
    
    with torch.no_grad():
        for images, labels in test_loader:
            # Forward pass
            conv1 = conv2d(images, conv1_weights, stride=1, padding=1)
            conv1 = relu(batch_norm(conv1))
            conv1 = max_pool2d(conv1, kernel_size=2)
            
            conv2 = conv2d(conv1, conv2_weights, stride=1, padding=1)
            conv2 = relu(batch_norm(conv2))
            conv2 = max_pool2d(conv2, kernel_size=2)
            
            conv3 = conv2d(conv2, conv3_weights, stride=1, padding=1)
            conv3 = relu(batch_norm(conv3))
            conv3 = max_pool2d(conv3, kernel_size=2)
            
            out = conv3.reshape(images.shape[0], -1).T
            out = relu(batch_norm(torch.mm(W1, out) + b1.view(-1, 1)))
            out = relu(batch_norm(torch.mm(W2, out) + b2.view(-1, 1)))
            out = relu(batch_norm(torch.mm(W3, out) + b3.view(-1, 1)))
            out = relu(batch_norm(torch.mm(W4, out) + b4.view(-1, 1)))
            out = softmax(torch.mm(W5, out) + b5.view(-1, 1))
            
            predicted = torch.argmax(out, dim=0)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
    
    accuracy = (correct / total) * 100
    print(f"Test Accuracy: {accuracy:.2f}%")
    return accuracy

# Training
learning_rate = 0.01
epochs = 50

start_time = time.time()
cost_values, test_acc_history, train_acc_history = network_with_conv(None, None, learning_rate, epochs)
total_time = time.time() - start_time

# Plotting
plt.figure(figsize=(15, 5))

# Plot training loss
plt.subplot(1, 2, 1)
plt.plot(range(len(cost_values)), cost_values)
plt.xlabel('Iterations')
plt.ylabel('Cost Value')
plt.title('Cost Value over Iterations')

# Plot accuracies
plt.subplot(1, 2, 2)
plt.plot(range(epochs), train_acc_history, label='Training Accuracy')
plt.plot(range(epochs), test_acc_history, label='Test Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy (%)')
plt.title('Training and Test Accuracy over Epochs')
plt.legend()

plt.tight_layout()
plt.savefig("training_curves.png")

print(f"\nTraining completed in {total_time:.2f} seconds")
print(f"Final cost value: {cost_values[-1]:.6f}")
print(f"Final training accuracy: {train_acc_history[-1]:.2f}%")
print(f"Final test accuracy: {test_acc_history[-1]:.2f}%")
