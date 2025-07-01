from torch.utils.data import DataLoader
from torchvision import datasets, transforms

def parse_transforms(transform_list):
    transform_ops = []
    for item in transform_list:
        if isinstance(item, dict):
            for key, value in item.items():
                if key == "Resize":
                    transform_ops.append(transforms.Resize(tuple(value)))
                elif key == "ToTensor" and value:
                    transform_ops.append(transforms.ToTensor())
                elif key == "Normalize":
                    # Используем стандартное std, если его нет в конфиге
                    std = [0.229, 0.224, 0.225]
                    transform_ops.append(transforms.Normalize(mean=value, std=std))
    return transforms.Compose(transform_ops)

def load_data(config):
    train_transforms = parse_transforms(config['transforms']['train_transforms'])
    test_transforms = parse_transforms(config['transforms']['test_transforms'])

    train_dataset = datasets.ImageFolder(config['train_data_path'], transform=train_transforms)
    test_dataset = datasets.ImageFolder(config['test_data_path'], transform=test_transforms)

    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config['batch_size'], shuffle=False)

    return train_loader, test_loader
