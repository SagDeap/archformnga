

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image
import random

class ShufflerNet(nn.Module):
    def __init__(self, in_channels=1):
        super(ShufflerNet, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.shuffle1 = self._make_shuffle_layer(64, groups=4)
        self.shuffle2 = self._make_shuffle_layer(64, groups=4)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.shuffle3 = self._make_shuffle_layer(64, groups=4)
        self.conv3 = nn.Conv2d(64, in_channels, kernel_size=3, stride=1, padding=1)

    def _make_shuffle_layer(self, channels, groups):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels),
            nn.PReLU(),
            nn.Conv2d(channels, channels, kernel_size=1, stride=1),
            nn.BatchNorm2d(channels),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.conv1(x)
        x = self.shuffle1(x)
        x = self.shuffle2(x)
        x = self.conv2(x)
        x = self.shuffle3(x)
        x = self.conv3(x)

        return x


class Discriminator(nn.Module):
    def __init__(self, in_channels=1):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(128 * 96 * 128, 1)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        return x


def shuffle_tiles(output, reference_image, output_path, epoch, iteration):
    shuffled_tiles = random.sample(output.unbind(dim=1), len(output.unbind(dim=1)))
    shuffled_image = torch.cat(shuffled_tiles, dim=2)

    
    shuffled_image_path = os.path.join(output_path, f"shuffled_epoch_{epoch}_iteration_{iteration}.png")
    save_image(shuffled_image, shuffled_image_path)

    print(f"Shuffled image saved at {shuffled_image_path}", flush=True)

    return shuffled_image


def train_epoch(generator, discriminator, dataloader, criterion_reconstruction, criterion_gan, optimizer_gen, optimizer_disc, epoch, output_path, writer):
    generator.train()
    discriminator.train()

    for batch_idx, batch_data in enumerate(dataloader):
        data, prac_data = batch_data[0], batch_data[1]

        optimizer_gen.zero_grad()
        optimizer_disc.zero_grad()

        try:
            fake_output = generator(data)

            # Adversarial loss (GAN loss)
            fake_preds = discriminator(fake_output)
            real_labels = torch.ones_like(fake_preds)
            gan_loss = criterion_gan(fake_preds, real_labels)

            
            reconstruction_loss = criterion_reconstruction(fake_output, data)

           
            total_loss = reconstruction_loss + lambda_gan * gan_loss

            print(f"Iteration {batch_idx + 1} | Loss: {total_loss.item()}", flush=True)
            print("---")

            if batch_idx % 5 == 0:
                intermediate_iteration_folder = os.path.join(output_path, "intermediate")
                os.makedirs(intermediate_iteration_folder, exist_ok=True)

                intermediate_image = fake_output[0].detach().unsqueeze(0)

                
                intermediate_iteration_path = os.path.join(intermediate_iteration_folder,
                                                            f"epoch_{epoch + 1}_iteration_{batch_idx + 1}.png")
                save_image(intermediate_image, intermediate_iteration_path, format='png')
                print(f"Intermediate result saved at: {intermediate_iteration_path}", flush=True)
                print("---")

        except Exception as e:
            print(f"Iteration {batch_idx + 1} failed with error: {e}")
            print(f"Full traceback: {e}", flush=True)

        total_loss.backward()
        optimizer_gen.step()

    
    save_model(generator, discriminator, epoch, output_path)

   
    writer.add_scalar("Total Loss", total_loss.item(), epoch)


def save_model(generator, discriminator, epoch, output_path):
    gen_model_folder = os.path.join(output_path, "generator")
    disc_model_folder = os.path.join(output_path, "discriminator")

    os.makedirs(gen_model_folder, exist_ok=True)
    os.makedirs(disc_model_folder, exist_ok=True)

    gen_model_path = os.path.join(gen_model_folder, f"gen_model_epoch_{epoch + 1}.pth")
    disc_model_path = os.path.join(disc_model_folder, f"disc_model_epoch_{epoch + 1}.pth")

    torch.save(generator.state_dict(), gen_model_path)
    torch.save(discriminator.state_dict(), disc_model_path)

    print(f"Models saved at epoch {epoch + 1}:")
    print(f"Generator: {gen_model_path}")
    print(f"Discriminator: {disc_model_path}")
