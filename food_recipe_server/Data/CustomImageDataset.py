import os
import warnings
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image
from PIL import ImageFile

warnings.filterwarnings("ignore", message="Palette images with Transparency expressed in bytes should be converted to RGBA images")
warnings.filterwarnings("ignore", category=UserWarning, module="PIL")

ImageFile.LOAD_TRUNCATED_IMAGES = True

def convert_to_rgb(img):
    return img.convert("RGB") if img.mode != "RGB" else img

class CustomImageDataset(Dataset):
    def __init__(self, image_folder, transform=None):
        self.image_paths = []
        self.labels = []
        self.transform = transform
        self.class_to_idx = {}  # 클래스명과 인덱스 매핑 딕셔너리
        self.classes = []       # 클래스 이름 리스트

        # 이미지 파일 경로 및 레이블 추출
        for root, _, files in os.walk(image_folder):
            for file in files:
                if file.endswith(('.jpg', '.jpeg', '.png')):  # 이미지 파일 필터링
                    # 이미지 경로 추가
                    image_path = os.path.join(root, file)
                    self.image_paths.append(image_path)

                    # 파일 경로에서 마지막 폴더명 추출
                    class_name = os.path.basename(os.path.dirname(image_path))  # 마지막 폴더명
                    if class_name not in self.class_to_idx:
                        self.class_to_idx[class_name] = len(self.classes)
                        self.classes.append(class_name)

                    # 클래스 인덱스 추가
                    self.labels.append(self.class_to_idx[class_name])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        try:
            image = Image.open(image_path).convert("RGB")
        except (OSError, IOError):
            print(f"이미지를 열 수 없습니다: {image_path}")
            return None, None  # 에러 발생 시 기본값 반환
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, int(label)  # 이미지와 정수형 레이블 반환

    @classmethod
    def create_image_dataset(cls, directory):
        transform = transforms.Compose([
            transforms.Lambda(convert_to_rgb),
            transforms.Resize((244,244)),
            transforms.ToTensor(),
            transforms.RandomRotation(10),  # 이미지를 무작위로 10도 내외로 회전
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),  # 밝기, 대비, 채도 조정
            transforms.RandomAffine(degrees=15, translate=(0.1, 0.1)),  # 무작위 이동 및 회전
            transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

        dataset = CustomImageDataset(image_folder=directory, transform=transform)

        return dataset