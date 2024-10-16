# @title data_loader.py

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import os
import shutil

class MangaDataset(Dataset):
    prac_folder = ""

    def __init__(self, image_folder, transform=None):
        self.image_folder = image_folder
        self.image_paths = [os.path.join(image_folder, img) for img in os.listdir(image_folder)]
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        img = Image.open(img_path)

        prac_idx = os.path.splitext(os.path.basename(img_path))[0]
        prac_img_path = os.path.join(self.prac_folder, f"{prac_idx}.jpg")
        prac_img = Image.open(prac_img_path)

        
        temp_folder = self.create_tiles(img_path)

       
        tile_paths = [os.path.join(temp_folder, tile) for tile in os.listdir(temp_folder)]

        
        tiles = [self.transform(Image.open(tile)) for tile in tile_paths]

        
        shutil.rmtree(temp_folder)

        
        stacked_tiles = torch.stack(tiles, dim=0)

        
        stacked_tiles = stacked_tiles.unsqueeze(0).permute(0, 2, 1, 3, 4)

        return stacked_tiles, self.transform(prac_img)

    def create_tiles(image, output_folder):
      img_name = os.path.splitext(os.path.basename(image))[0]
      tile_width = 96
      tile_height = 128
      img = Image.open(image)
      width, height = img.size
      width_n = width // tile_width
      height_n = height // tile_height

     
      temp_folder = os.path.join(output_folder, f"temp_tiles_{img_name}")
      os.makedirs(temp_folder, exist_ok=True)

      for i in range(height_n):
          for n in range(width_n):
              crop_img = img.crop((tile_width * n, tile_height * i, tile_width * (n + 1), tile_height * (i + 1)))
              tile_path = os.path.join(temp_folder, f"{img_name}_{i}_{n}.png")
              crop_img.save(tile_path)

      return temp_folder

def get_dataloader(dataset, batch_size=1, shuffle=True):
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)