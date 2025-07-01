from models.model import FaceResNet  # Импорт модели
from train import train_model  # Импорт функции тренировки
from test import evaluate_model  # Импорт функции оценки
from utils.image_utils import load_data  # Импорт функции загрузки данных
import yaml

def load_config(path="config.yaml"):
    """
    Функция для загрузки конфигурации из YAML файла.
    """
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)

if __name__ == "__main__":
    # Загружаем конфигурацию
    config = load_config("config.yaml")

    # Загружаем данные
    train_loader, test_loader = load_data(config)  # Загружаем тренировочные и тестовые данные

    # Создаем модель
    model = FaceResNet(config)  # Передаем конфигурацию в модель, если это необходимо

    # Обучаем модель
    train_model(model, train_loader, test_loader, config)  # Передаем test_loader для использования в процессе обучения (например, для валидации)

    # Тестируем модель
    evaluate_model(model, test_loader)  # Оценка на тестовых данных
