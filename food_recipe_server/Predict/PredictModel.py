from torchvision import transforms
from PIL import Image
from torch.amp import GradScaler, autocast
import torch

def predict_image(model, image_path, class_names):
    model.eval()  # 모델을 평가 모드로 설정

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        with autocast(device_type='cuda'):
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)

    return class_names[predicted.item()]