import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# Convert images to tensors
transform = transforms.Compose([transforms.ToTensor()])

# Load training dataset
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
# DISTILLATION TRAINING
# -------------------------
optimizer_student = optim.Adam(student_model.parameters(), lr=0.001)

temperature = 3.0

for epoch in range(3):
    for images, labels in trainloader:

        optimizer_student.zero_grad()

        # Teacher predictions
        with torch.no_grad():
            teacher_outputs = teacher_model(images)

        # Student predictions
        student_outputs = student_model(images)

        # Soft targets
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
# TEST STUDENT ACCURACY
# -------------------------
correct = 0
total = 0

with torch.no_grad():
    for images, labels in testloader:

        outputs = student_model(images)
        _, predicted = torch.max(outputs.data, 1)

        total += labels.size(0)
        correct += (predicted == labels).sum().item()

student_accuracy = 100 * correct / total

print("Student Model Accuracy:", student_accuracy, "%")

# Count parameters in teacher model
teacher_params = sum(p.numel() for p in teacher_model.parameters())

# Count parameters in student model
student_params = sum(p.numel() for p in student_model.parameters())

print("Teacher parameters:", teacher_params)
print("Student parameters:", student_params)




