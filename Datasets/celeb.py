from torch.utils.data import Dataset
from PIL import Image
import os
from pathlib import Path
from utilities.utils import standardize
from torchvision import transforms
import os.path


class CelebA:
    def __init__(self, dataroot, dim=1024):
        transform = transforms.Compose([transforms.Resize([dim, dim]), transforms.ToTensor(), standardize])
        path = Path(dataroot) / "data" / "CELEB"
        self.train_dataset = CELEBDataset(path, transform=transform, eval=False)
        self.test_dataset = CELEBDataset(path, transform=transform, eval=True)
        self.imDim = (3, dim, dim)
        self.num_classes = 2


class CELEBDataset(Dataset):
    def __init__(self, img_path, transform, eval):
        if not eval:
            self.img_path = os.path.join(img_path, "train\\")
        else:
            self.img_path = os.path.join(img_path, "eval\\")
        self.ims = os.listdir(self.img_path)
        self.transform = transform

    def __getitem__(self, index):
        img_name = os.path.join(self.img_path, self.ims[index])
        img = Image.open(img_name)
        img = self.transform(img)
        return img, 1

    def __len__(self):
        return len(self.ims)
