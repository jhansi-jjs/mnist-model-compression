import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.nn.utils.prune as prune
from sklearn.metrics import precision_score, recall_score
import numpy as np
import time
import matplotlib.pyplot as plt

# -------------------------
# DATA
# -------------------------

transform = transforms.Compose([transforms.ToTensor()])

trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

testset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True)
testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False)

print("Training samples:", len(trainset))
print("Test samples:", len(testset))

# -------------------------
# MODEL
# -------------------------

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):

        x = x.view(-1, 28*28)

        x = F.relu(self.fc1(x))

        x = self.fc2(x)

        return x

# -------------------------
# TRAINING FUNCTION
# -------------------------

def train_model(model):

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(3):

        for images, labels in trainloader:

            optimizer.zero_grad()

            outputs = model(images)

            loss = criterion(outputs, labels)

            loss.backward()

            optimizer.step()

        print(f"Epoch {epoch+1} complete")

# -------------------------
# EVALUATION FUNCTION
# -------------------------

def evaluate_model(model):

    model.eval()

    all_preds = []
    all_labels = []
    times = []

    with torch.no_grad():

        for images, labels in testloader:

            start = time.time()

            outputs = model(images)

            end = time.time()

            _, predicted = torch.max(outputs, 1)

            times.append(end-start)

            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)

    precision = precision_score(all_labels, all_preds, average="macro")

    recall = recall_score(all_labels, all_preds, average="macro")

    avg_time = np.mean(times)
    std_time = np.std(times)

    params = sum(p.numel() for p in model.parameters())

    return accuracy, precision, recall, avg_time, std_time, params

# -------------------------
# SPARSITY FUNCTION
# -------------------------

def calculate_sparsity(layer):

    weight = layer.weight.detach().cpu().numpy()

    total_weights = weight.size

    zero_weights = (weight == 0).sum()

    sparsity = 100 * zero_weights / total_weights

    return sparsity

# -------------------------
# WEIGHT DISTRIBUTION PLOT
# -------------------------

def plot_weight_distribution(layer, title):

    weights = layer.weight.detach().cpu().numpy().flatten()

    plt.figure()

    plt.hist(weights, bins=50)

    plt.title(title)

    plt.xlabel("Weight value")

    plt.ylabel("Frequency")

    plt.show()

# -------------------------
# BASELINE MODEL
# -------------------------

baseline_model = Net()

print("\nTraining Baseline Model")

train_model(baseline_model)

baseline_results = evaluate_model(baseline_model)

plot_weight_distribution(baseline_model.fc1, "Baseline Weight Distribution")

# -------------------------
# MAGNITUDE PRUNING
# -------------------------

magnitude_model = Net()

magnitude_model.load_state_dict(baseline_model.state_dict())

prune.l1_unstructured(magnitude_model.fc1, name="weight", amount=0.5)

print("\nMagnitude pruning applied")

magnitude_results = evaluate_model(magnitude_model)

mag_sparsity = calculate_sparsity(magnitude_model.fc1)

plot_weight_distribution(magnitude_model.fc1, "Magnitude Pruned Weight Distribution")

# -------------------------
# RANDOM PRUNING
# -------------------------

random_model = Net()

random_model.load_state_dict(baseline_model.state_dict())

prune.random_unstructured(random_model.fc1, name="weight", amount=0.5)

print("\nRandom pruning applied")

random_results = evaluate_model(random_model)

rand_sparsity = calculate_sparsity(random_model.fc1)

plot_weight_distribution(random_model.fc1, "Random Pruned Weight Distribution")

# -------------------------
# PRINT RESULTS
# -------------------------

def print_results(name, results, sparsity=None):

    acc, prec, rec, avg, std, params = results

    print("\n==============================")
    print(name)
    print("==============================")

    print("Parameters:", params)
    print("Accuracy:", acc)
    print("Macro Precision:", prec)
    print("Macro Recall:", rec)

    print("Average inference time:", avg)
    print("Average + std:", avg + std)
    print("Average - std:", avg - std)

    if sparsity is not None:
        print("Sparsity:", sparsity, "%")

print_results("BASELINE MODEL", baseline_results, 0)
print_results("MAGNITUDE PRUNED MODEL", magnitude_results, mag_sparsity)
print_results("RANDOM PRUNED MODEL", random_results, rand_sparsity)