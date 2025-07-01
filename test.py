import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from torchvision import transforms, datasets
from torch.utils.data import DataLoader
from PIL import Image
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import yaml
import os


# Загрузка параметров из config.yaml
with open("config.yaml", "r") as f:
    config = yaml.safe_load(f)

image_size = tuple(config["image_size"])
batch_size = config["batch_size"]

# Преобразования (нормализация по ImageNet)
transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# Подготовка датасета и загрузчика
test_dataset = datasets.ImageFolder(config["test_data_path"], transform=transform)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Загрузка модели
from models.model import FaceResNet
model = FaceResNet(num_classes=2)
model.load_state_dict(torch.load("models/model.pth", map_location="cpu"))

# Оценка на тестовых данных
from sklearn.metrics import classification_report

def evaluate_model(model, test_loader, class_names=["не лицо", "лицо"]):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to("cpu"), labels.to("cpu")
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            all_labels.extend(labels.numpy())
            all_preds.extend(predicted.numpy())

    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='weighted')
    recall = recall_score(all_labels, all_preds, average='weighted')
    f1 = f1_score(all_labels, all_preds, average='weighted')
    report = classification_report(all_labels, all_preds, target_names=class_names)

    print(f"\nAccuracy: {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}")
    print("\nClassification Report:\n", report)

    # Строим confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(5,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.show()

    # Сохранение в файл
    with open("test_results.txt", "w", encoding="utf-8") as f:
        f.write(f"Accuracy: {accuracy:.4f}\n") # верные от всех
        f.write(f"Precision: {precision:.4f}\n") # верные среди лиц
        f.write(f"Recall: {recall:.4f}\n") # найденные лица среди настоящих лиц
        f.write(f"F1 Score: {f1:.4f}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
    print("Отчёт сохранён в test_results.txt")


# Предсказание одиночного изображения
def predict(model, image_path, class_names):
    model.eval()
    transform = transforms.Compose([
        transforms.Resize(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    image = Image.open(image_path).convert('RGB')
    image = transform(image).unsqueeze(0)

    with torch.no_grad():
        output = model(image)
        probs = torch.softmax(output, dim=1)
        confidence, predicted_class = torch.max(probs, 1)

    class_name = class_names[predicted_class.item()]
    print(f"Предсказано: {class_name} (уверенность: {confidence.item():.2f}) для изображения: {image_path}")

# Запуск
evaluate_model(model, test_loader)

# Предсказание одиночного изображения с проверкой пути
def predict_multiple(model, class_names):
    while True:
        # Запрос пути к изображению у пользователя
        image_path = input("Введите путь к изображению (или 'stop' для завершения), начиная путь с data...: ").strip('"')

        # Условие для выхода из цикла
        if image_path.lower() == 'stop':
            print("Завершаем программу...")
            break
        
        # Проверка, существует ли файл по указанному пути
        if os.path.exists(image_path):
            # Если файл существует, выполняем предсказание
            predict(model, image_path, class_names)
        else:
            # Если путь некорректен, выводим сообщение и повторяем ввод
            print(f"Путь '{image_path}' не существует. Пожалуйста, введите правильный путь.")


# Классы
#class_names = ["лицо", "не лицо"]
class_names = test_dataset.classes  # Это гарантированно верный порядок

# Получить предсказания для нескольких изображений
predict_multiple(model, class_names)
