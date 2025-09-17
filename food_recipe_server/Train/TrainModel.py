import torch
import torch.nn as nn
import torch.optim as optim
from Data.CustomImageDataset import CustomImageDataset
from Validate.ValidateModel import validate_model
from torch.amp import GradScaler, autocast
from torch.utils.data import DataLoader, Subset
from sklearn.model_selection import KFold


def train_model(model, dataset_path, model_save_path, learning_rate=0.001, batch_size=32, epochs=50, patience=5, num_workers=0):

    dataset = CustomImageDataset.create_image_dataset(dataset_path)

    # 손실 함수 및 옵티마이저 정의
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    scaler = GradScaler(init_scale=1024, growth_factor=2.0, enabled=True)
    kf = KFold(n_splits=5, shuffle=True, random_state=42)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model.to(device)

    for fold, (train_idx, val_idx) in enumerate(kf.split(dataset)):
        print(f'\nFold {fold + 1}')

        train_subset = Subset(dataset, train_idx)
        val_subset = Subset(dataset, val_idx)

        train_loader = DataLoader(train_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)
        val_loader = DataLoader(val_subset, batch_size=batch_size, shuffle=True, num_workers=num_workers)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            model.train()

            for i, (inputs, labels) in enumerate(train_loader, 0):
                inputs, labels = inputs.to(device), labels.to(device).long()
                optimizer.zero_grad()

                with autocast(device_type='cuda'):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()

                # 통계 출력
                print(f'\r[{epoch + 1}, {i + 1}] loss: {loss.item():.3f}', end='')

            scheduler.step()

            # 검증 단계
            val_loss, val_acc = validate_model(model, val_loader, criterion, device)
            print()
            print(f'Epoch {epoch + 1} - Validation Loss: {val_loss:.4f}, Validation Accuracy: {val_acc:.2f}%')

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                torch.save(model.state_dict(), model_save_path)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("조기 중지 조건 충족. 학습 종료.")
                    break