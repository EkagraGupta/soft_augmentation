from models.wideresnet import wideresnet28d3, wideresnet28d0
import torch 
from utils import get_test_dataloader

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mean = [x / 255.0 for x in [125.3, 123.0, 113.9]]
std = [x / 255.0 for x in [63.0, 62.1, 66.7]]

cifar10_testloader = get_test_dataloader(mean, std, batch_size=128, task='cifar10', shuffle=False)

model = wideresnet28d3(num_classes=10)
# Load the model
PATH = '/home/ekagra/personal/soft_augmentation/cifar/models/trained/wideresnet28-cifar10-200-regular.pth'
state_dict = torch.load(PATH, map_location=torch.device('cpu'))
# Create a new state dict without 'module.' prefix
new_state_dict = {}
for k, v in state_dict.items():
    new_key = k.replace('module.', '')  # Remove the 'module.' prefix
    new_state_dict[new_key] = v
model.load_state_dict(state_dict=new_state_dict, strict=True)
model = model.to(device)
# print(model)

correct, total = 0.0, 0.0
model.eval()

with torch.no_grad():
    for data in cifar10_testloader:
        images, labels = data
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

print('Accuracy of the network on the test images: %d %%' % (100 * correct / total))