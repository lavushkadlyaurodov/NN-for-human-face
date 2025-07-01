import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

def prepare_data(train_dir, val_dir, test_dir, batch_size=32):

    # Проверка существования директорий
    for dir_path in [train_dir, val_dir, test_dir]:
        if not os.path.isdir(dir_path):
            raise ValueError(f"Directory does not exist: {dir_path}")

    # Трансформации для тренировочных данных (с аугментациями)
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(10),
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Трансформации для валидации и теста (без аугментаций)
    test_transform = transforms.Compose([
        transforms.Resize(128),
        transforms.CenterCrop(112),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Загрузка датасетов
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=test_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)
    
    # Получаем имена классов
    class_names = train_dataset.classes
    
    # Создаем DataLoader
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    
    return train_loader, val_loader, test_loader, class_names

