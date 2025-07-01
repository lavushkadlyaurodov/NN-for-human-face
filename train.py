import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import copy
from processed_data import prepare_data  # Импортируем функцию для подготовки данных
from models.model import FaceResNet  # Модель, которая обучается
import yaml
import time

def train_model(model, train_loader, val_loader, config):
    device = "cpu"
    model.to(device)

    # Подбор весов классов вручную (примерно 4300 лиц и 4300 не лиц)
    weights = torch.tensor([1.0, 1.0], dtype=torch.float)  # [face, non face]
    weights = weights.to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)

    optimizer = optim.Adam(model.parameters(), lr=config['learning_rate'])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', patience=2, factor=0.5, verbose=True)

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = float('inf')

    # Перебор эпох
    for epoch in range(config['epochs']):
        print(f"\nEpoch {epoch+1}/{config['epochs']} — Training...")
        epoch_start = time.time()

        model.train()
        running_loss = 0.0
        running_corrects = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, preds = torch.max(outputs, 1)
            running_corrects += torch.sum(preds == labels.data)

        epoch_loss = running_loss / len(train_loader.dataset)
        epoch_acc = running_corrects.double() / len(train_loader.dataset)
        
        print("Validating...")
        val_start = time.time()

        # Валидация
        model.eval()
        val_loss = 0.0
        val_corrects = 0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * inputs.size(0)
                _, preds = torch.max(outputs, 1)
                val_corrects += torch.sum(preds == labels.data)

        val_loss /= len(val_loader.dataset)
        val_acc = val_corrects.double() / len(val_loader.dataset)
        scheduler.step(val_loss)

        # Итог эпохи
        epoch_time = time.time() - epoch_start
        val_time = time.time() - val_start

        print(f"Epoch {epoch+1}: "
              f"Train Loss: {epoch_loss:.4f}, Acc: {epoch_acc:.4f} | "
              f"Val Loss: {val_loss:.4f}, Acc: {val_acc:.4f} | "
              f"Time: {epoch_time:.2f}s (Val: {val_time:.2f}s)")

        # Сохранение модели с наименьшей валидационной потерей
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_wts = copy.deepcopy(model.state_dict())
            torch.save(model.state_dict(), 'models/best_model.pth')  # Сохраняем модель
            print("Saved new best model.")

    print("Training complete. Best val loss: {:.4f}".format(best_val_loss))
    model.load_state_dict(best_model_wts)  # Загружаем лучшие веса модели
    
    # Сохраняем лучшую модель под стандартным именем
    torch.save(model.state_dict(), "models/model.pth")
    return model

# Запуск обучения
if __name__ == "__main__":
    # Чтение конфигурации из файла config.yaml
    with open("config.yaml", 'r') as config_file:
        config = yaml.safe_load(config_file)

    # Подготовка данных с использованием параметров из config.yaml
    train_loader, val_loader, test_loader, class_names = prepare_data(
        train_dir=config['train_data_path'],
        val_dir=config['train_data_path'],  # Нет пути для валидации, поэтому используем тренировочные данные
        test_dir=config['test_data_path'],
        batch_size=config['batch_size']
    )

    # Инициализация модели с нужным количеством классов
    model = FaceResNet(num_classes=2)  # Укажи нужное количество классов

    # Обучение модели
    train_model(model, train_loader, val_loader, config)
