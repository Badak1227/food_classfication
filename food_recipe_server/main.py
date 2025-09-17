import os
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pandas as pd


from Transmission.FlaskServer import *
from Models.Efficientnet import EfficientNetModel
from Train.TrainModel import *
from RecipeAPI.Recipe import fetch_recipes
from Data.CustomImageDataset import *

# 경로 설정
dataset_path = './dataset'
model_save_path = '모델 저장 경로'
list_path = '데이터 클래스들이 정의된 엑셀 파일 경로'

api_url = "openapi.foodsafetykorea.go.kr의 인증키 경로"

# 하이퍼파라미터 설정
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

# 클래스 이름, 클래스 수 엑셀파일에서 읽어오기
data = pd.read_excel(list_path)
class_names = data['Class Name'].tolist()
num_classes = len(class_names)

model = EfficientNetModel(num_classes, model_save_path, pretrained=False)

# API 데이터 가져오기
recipe_data = fetch_recipes(api_url)
app = create_app(model, class_names, recipe_data)

# 모델 학습 시 현재 파일 실행
if __name__ == '__main__':

    dataset = CustomImageDataset.create_image_dataset(dataset_path)

    TrainModel.train_model(model, dataset_path, model_save_path, learning_rate=LEARNING_RATE, batch_size=BATCH_SIZE, epochs=EPOCHS, patience=5, num_workers=6)



