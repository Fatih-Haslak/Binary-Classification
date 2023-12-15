import os
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
from torch.utils.data import DataLoader, random_split

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class CustomDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.classes = os.listdir(root_dir)
        self.class_to_idx = {cls: idx for idx, cls in enumerate(self.classes)}
        self.data = self.load_data()
        self.transform = transform

    def load_data(self):
        data = []
        for class_name in self.classes:
            class_path = os.path.join(self.root_dir, class_name)
            for filename in os.listdir(class_path):
                if filename.endswith(".jpg"):
                    img_path = os.path.join(class_path, filename)
                    label = (self.class_to_idx[class_name])
                    data.append((img_path, label))
        return data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        img_path, label = self.data[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)
            
        return image, label

class DogsAndCatsDataLoader:
    def __init__(self, root_folder):
        self.root_folder = root_folder
        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

    def load_data(self):
        custom_dataset = CustomDataset(self.root_folder, transform=self.transform)

        # Veri setini train ve validation olarak bölelim
        total_size = len(custom_dataset)
        train_size = int(0.8 * total_size)  # Örnek olarak %80 train, %20 validation
        val_size = total_size - train_size

        train_dataset, val_dataset = random_split(custom_dataset, [train_size, val_size])

        # Dataloader'ları oluşturun
        train_dataloader = DataLoader(train_dataset, batch_size=4, shuffle=True, num_workers=4)
        val_dataloader = DataLoader(val_dataset, batch_size=4, shuffle=False, num_workers=4)

        return train_dataloader, val_dataloader
