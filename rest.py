import torch
from torchvision.transforms import functional as F
from PIL import Image
import torchvision.transforms as T

image_labels = {
    'combat':0,
    'humanitarianaid':3,
    'militaryvehicles':4,
    'fire':2,
    'destroyedbuilding':1 
}

model = torch.load('rest.pt')
model.eval()

image_path = 'New folder/fire1.jpeg'
image = Image.open(image_path)

transform = T.Compose([T.ToTensor()])

input_image = transform(image).unsqueeze(0)  # Add batch dimension


with torch.no_grad():
    output = model(input_image)

predicted_label = output.argmax().item()

#print("Predicted class label:", predicted_label)
print(list(image_labels.keys())[list(image_labels.values()).index(predicted_label)])

output_np = output.numpy()