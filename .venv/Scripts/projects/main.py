import os
import torch
import pandas as pd

from Models.Efficientnet import EfficientNetModel
from Predict.PredictModel import *
from Train.TrainModel import *

# 경로 설정
dataset_path = 'dataset_path'  # 이미지 하위폴더들이 있는 경로
model_save_path = 'model_save_path'  # 모델 저장 경로
list_path = 'list.xlsx_path'

# 하이퍼파라미터 설정
BATCH_SIZE = 32
EPOCHS = 50
LEARNING_RATE = 0.001

if __name__ == "__main__":
    data = pd.read_excel(list_path)
    class_names = data['Class Name'].tolist()
    num_classes = len(class_names)

    model = EfficientNetModel(num_classes, model_save_path, pretrained=True)

    train_model(model, dataset_path, model_save_path, batch_size=BATCH_SIZE, learning_rate=LEARNING_RATE, epochs=EPOCHS, patience = 5, num_workers = 6)

    for root,_,files in os.walk("C:/picture/이미지"):
        for img in files:
            print(img)
            print(predict_image(model, os.path.join(root, img), class_names))