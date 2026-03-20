import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
import numpy as np
import time

# -------------------------
# DATA TRANSFORM
# -------------------------
transform = transforms.Compose([transforms.ToTensor()])

# -------------------------
# LOAD TRAIN DATASET
# -------------------------
trainset = torchvision.datasets.MNIST(
    root='./data',
    train=True,
    download=True,
    transform=transform
)

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

# -------------------------
# TEACHER NETWORK
# -------------------------
class TeacherNet(nn.Module):
    def __init__(self):
        super(TeacherNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

teacher_model = TeacherNet()

# -------------------------
# STUDENT NETWORK
# -------------------------
class StudentNet(nn.Module):
    def __init__(self):
        super(StudentNet, self).__init__()
        self.fc1 = nn.Linear(28*28, 32)
        self.fc2 = nn.Linear(32, 10)

    def forward(self, x):
        x = x.view(-1, 28*28)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

student_model = StudentNet()

# -------------------------
# TRAIN TEACHER MODEL
# -------------------------
criterion = nn.CrossEntropyLoss()
optimizer_teacher = optim.Adam(teacher_model.parameters(), lr=0.001)

for epoch in range(3):
    for images, labels in trainloader:

        optimizer_teacher.zero_grad()
        outputs = teacher_model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer_teacher.step()

    print(f"Teacher Epoch {epoch+1} complete")

print("Teacher training finished")

# -------------------------
# KNOWLEDGE DISTILLATION
# -------------------------
optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)
temperature = 3.0

for epoch in range(3):
    for images, labels in trainloader:

        optimizer_student.zero_grad()

        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        student_outputs = student_model(images)

        soft_teacher = F.softmax(teacher_outputs / temperature, dim=1)
        soft_student = F.log_softmax(student_outputs / temperature, dim=1)

        loss = F.kl_div(soft_student, soft_teacher, reduction='batchmean')

        loss.backward()
        optimizer_student.step()

    print(f"Student Epoch {epoch+1} complete")

print("Student training finished")

# -------------------------
# LOAD TEST DATASET
# -------------------------
testset = torchvision.datasets.MNIST(
    root='./data',
    train=False,
    download=True,
    transform=transform
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False
)

# -------------------------
# EVALUATION FUNCTION
# -------------------------
def evaluate_model(model):

    model.eval()

    all_preds = []
    all_labels = []
    inference_times = []

    with torch.no_grad():
        for images, labels in testloader:

            start = time.time()

            outputs = model(images)

            end = time.time()

            _, predicted = torch.max(outputs, 1)

            inference_times.append(end - start)

            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100 * sum([p == l for p, l in zip(all_preds, all_labels)]) / len(all_labels)

    precision = precision_score(all_labels, all_preds, average="macro")
    recall = recall_score(all_labels, all_preds, average="macro")

    avg_time = np.mean(inference_times)
    std_time = np.std(inference_times)

    return accuracy, precision, recall, avg_time, std_time


# -------------------------
# EVALUATE BOTH MODELS
# -------------------------
teacher_accuracy, teacher_precision, teacher_recall, teacher_avg, teacher_std = evaluate_model(teacher_model)

student_accuracy, student_precision, student_recall, student_avg, student_std = evaluate_model(student_model)

# -------------------------
# PARAMETER COUNT
# -------------------------
teacher_params = sum(p.numel() for p in teacher_model.parameters())
student_params = sum(p.numel() for p in student_model.parameters())

# -------------------------
# PRINT RESULTS
# -------------------------

print("\n========== TEACHER MODEL ==========")
print("Parameters:", teacher_params)
print("Accuracy:", teacher_accuracy)
print("Macro Precision:", teacher_precision)
print("Macro Recall:", teacher_recall)
print("Average inference time:", teacher_avg)
print("Average + std:", teacher_avg + teacher_std)
print("Average - std:", teacher_avg - teacher_std)

print("\n========== STUDENT MODEL ==========")
print("Parameters:", student_params)
print("Accuracy:", student_accuracy)
print("Macro Precision:", student_precision)
print("Macro Recall:", student_recall)
print("Average inference time:", student_avg)
print("Average + std:", student_avg + student_std)
print("Average - std:", student_avg - student_std)