import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import precision_score, recall_score
import time
import numpy as np

# -------------------------
# DATASET
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

trainloader = torch.utils.data.DataLoader(
    trainset,
    batch_size=64,
    shuffle=True
)

testloader = torch.utils.data.DataLoader(
    testset,
    batch_size=64,
    shuffle=False
)

# -------------------------
# TEACHER NETWORK
# -------------------------

class TeacherNet(nn.Module):

    def __init__(self):
        super(TeacherNet,self).__init__()

        self.fc1 = nn.Linear(28*28,128)
        self.fc2 = nn.Linear(128,10)

    def forward(self,x):

        x = x.view(-1,28*28)

        features = F.relu(self.fc1(x))

        output = self.fc2(features)

        return output, features


teacher_model = TeacherNet()

# -------------------------
# STUDENT NETWORK
# -------------------------

class StudentNet(nn.Module):

    def __init__(self):
        super(StudentNet,self).__init__()

        self.fc1 = nn.Linear(28*28,32)

        # projection layer
        self.projection = nn.Linear(32,128)

        self.fc2 = nn.Linear(32,10)

    def forward(self,x):

        x = x.view(-1,28*28)

        features = F.relu(self.fc1(x))

        projected_features = self.projection(features)

        output = self.fc2(features)

        return output, projected_features


student_model = StudentNet()

# -------------------------
# TRAIN TEACHER
# -------------------------

criterion = nn.CrossEntropyLoss()

optimizer_teacher = optim.Adam(
    teacher_model.parameters(),
    lr=0.001
)

for epoch in range(3):

    for images,labels in trainloader:

        optimizer_teacher.zero_grad()

        outputs,_ = teacher_model(images)

        loss = criterion(outputs,labels)

        loss.backward()

        optimizer_teacher.step()

    print(f"Teacher Epoch {epoch+1} complete")

print("Teacher training finished")

# -------------------------
# FEATURE DISTILLATION TRAINING
# -------------------------

optimizer_student = optim.Adam(
    student_model.parameters(),
    lr=0.001
)

mse_loss = nn.MSELoss()

for epoch in range(3):

    for images,labels in trainloader:

        optimizer_student.zero_grad()

        with torch.no_grad():
            teacher_outputs, teacher_features = teacher_model(images)

        student_outputs, student_features = student_model(images)

        loss_cls = criterion(student_outputs,labels)

        loss_feat = mse_loss(student_features,teacher_features)

        loss = loss_cls + loss_feat

        loss.backward()

        optimizer_student.step()

    print(f"Student Epoch {epoch+1} complete")

print("Student training finished")

# -------------------------
# EVALUATION FUNCTION
# -------------------------

def evaluate_model(model):

    model.eval()

    all_preds=[]
    all_labels=[]
    times=[]

    with torch.no_grad():

        for images,labels in testloader:

            start=time.time()

            outputs,_ = model(images)

            end=time.time()

            _,predicted = torch.max(outputs,1)

            times.append(end-start)

            all_preds.extend(predicted.numpy())
            all_labels.extend(labels.numpy())

    accuracy = 100*sum(
        [p==l for p,l in zip(all_preds,all_labels)]
    )/len(all_labels)

    precision = precision_score(
        all_labels,
        all_preds,
        average="macro"
    )

    recall = recall_score(
        all_labels,
        all_preds,
        average="macro"
    )

    avg_time = np.mean(times)
    std_time = np.std(times)

    return accuracy,precision,recall,avg_time,std_time

# -------------------------
# EVALUATE
# -------------------------

teacher_accuracy,teacher_precision,teacher_recall,teacher_avg,teacher_std = evaluate_model(teacher_model)

student_accuracy,student_precision,student_recall,student_avg,student_std = evaluate_model(student_model)

teacher_params = sum(
    p.numel() for p in teacher_model.parameters()
)

student_params = sum(
    p.numel() for p in student_model.parameters()
)

print("\n========== TEACHER MODEL ==========")
print("Parameters:",teacher_params)
print("Accuracy:",teacher_accuracy)
print("Macro Precision:",teacher_precision)
print("Macro Recall:",teacher_recall)
print("Average inference time:",teacher_avg)
print("Average + std:",teacher_avg+teacher_std)
print("Average - std:",teacher_avg-teacher_std)

print("\n========== FEATURE DISTILLED STUDENT ==========")
print("Parameters:",student_params)
print("Accuracy:",student_accuracy)
print("Macro Precision:",student_precision)
print("Macro Recall:",student_recall)
print("Average inference time:",student_avg)
print("Average + std:",student_avg+student_std)
print("Average - std:",student_avg-student_std)