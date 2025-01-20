import os
from torch.utils.data import Dataset
from PIL import Image
import s3fs

class S3ImageDataset(Dataset):
    def __init__(self, s3_root, transform=None, aws_key=None, aws_secret=None):
        self.s3 = s3fs.S3FileSystem(anon=False, key=aws_key, secret=aws_secret)
        self.s3_root = s3_root.rstrip("/")
        self.transform = transform

        self.image_paths = []
        self.class_to_idx = {}

        print(f"Fetching class directories from: {self.s3_root}")
        class_dirs = self.s3.ls(self.s3_root)
        for cls_dir in class_dirs:
            if self.s3.isdir(cls_dir):
                class_name = os.path.basename(cls_dir).replace(" ", "_")
                self.class_to_idx[class_name] = len(self.class_to_idx)
                cls_image_paths = self.s3.glob(f"{cls_dir}/**/*.jpg")
                self.image_paths.extend(cls_image_paths)

        if not self.image_paths:
            raise ValueError(f"No images found at {self.s3_root}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        with self.s3.open(image_path, "rb") as file:
            image = Image.open(file).convert("RGB")

        if self.transform:
            image = self.transform(image)

        label = os.path.basename(os.path.dirname(image_path)).replace(" ", "_")
        label = self.class_to_idx[label]

        return image, label
