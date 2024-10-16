# @title main.py

import os
import torch
import torch.optim as optim
from model import ShufflerNet, Discriminator, shuffle_tiles, train_epoch
from data_loader import MangaDataset, get_dataloader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
import sys
import random

def main():
    # Parameters
    num_epochs = 10
    lr = 0.001
    output_path = "your path here"
    input_data_path = "your path here"
    prac_folder_path = "your path here"
    lambda_gan = 1.0  

   
    generator = ShufflerNet()
    discriminator = Discriminator()

    
    pretrained_gen_path = os.path.join(output_path, "generator", "gen_model_epoch_0.pth")
    pretrained_disc_path = os.path.join(output_path, "discriminator", "disc_model_epoch_0.pth")

    if os.path.exists(pretrained_gen_path):
        print(f"Loading pre-trained generator from: {pretrained_gen_path}")
        generator.load_state_dict(torch.load(pretrained_gen_path))

    if os.path.exists(pretrained_disc_path):
        print(f"Loading pre-trained discriminator from: {pretrained_disc_path}")
        discriminator.load_state_dict(torch.load(pretrained_disc_path))

    else:
        print("No pre-trained model found. Training from scratch.")

    optimizer_gen = optim.Adam(generator.parameters(), lr=lr)
    optimizer_disc = optim.Adam(discriminator.parameters(), lr=lr)

    criterion_reconstruction = torch.nn.MSELoss()
    criterion_gan = torch.nn.BCEWithLogitsLoss()

    # Data loader
    transform = transforms.Compose([transforms.ToTensor()])
    MangaDataset.prac_folder = prac_folder_path
    train_dataset = MangaDataset(image_folder=input_data_path, transform=transform)
    train_dataloader = get_dataloader(train_dataset, batch_size=1, shuffle=True)

    
    writer = SummaryWriter()

    print("Entering the training loop...")

    for epoch in range(num_epochs):
        print(f"Epoch {epoch + 1}/{num_epochs}")
        total_loss = 0.0
        total_iterations = 0

        for _, batch_data in enumerate(train_dataloader):
            data, prac_data = batch_data[0], batch_data[1]
            
            optimizer_gen.zero_grad()

            try:
                
                img_path = os.path.basename(train_dataset.image_paths[total_iterations])
                print(f"Processing file: {img_path}")
                prac_img_path = os.path.join(MangaDataset.prac_folder, f"{os.path.splitext(img_path)[0]}.jpg")
                print(f"Corresponding reference image: {prac_img_path}")

               
               
                
                reference_image = prac_data

                fake_output = generator(data)

                
                fake_preds = discriminator(fake_output)
                real_labels = torch.ones_like(fake_preds)
                gan_loss = criterion_gan(fake_preds, real_labels)

                # Reconstruction loss
                reconstruction_loss = criterion_reconstruction(fake_output, data)

                # Total loss
                total_loss = reconstruction_loss + lambda_gan * gan_loss

                print(f"Iteration {total_iterations + 1} | Loss: {total_loss.item()}", flush=True)
                print("---")

                if total_loss.item() < 0.0001:
                    successful_iteration_folder = os.path.join(output_path, f"ep{epoch + 1}")
                    os.makedirs(successful_iteration_folder, exist_ok=True)

                    output_image = fake_output[0].detach().unsqueeze(0)

                    # Save the image in PNG format
                    successful_iteration_path = os.path.join(successful_iteration_folder, f"ep{epoch + 1}_{total_iterations + 1}.png")
                    save_image(output_image, successful_iteration_path, format='png')
                    print(f"Successful iteration saved at: {successful_iteration_path}", flush=True)
                    print("!!!!!!!!!!")

                if total_iterations % 5 == 0:
                    intermediate_iteration_folder = os.path.join(output_path, "intermediate")
                    os.makedirs(intermediate_iteration_folder, exist_ok=True)

                    intermediate_image = fake_output[0].detach().unsqueeze(0)

                    # Save the intermediate image in PNG format
                    intermediate_iteration_path = os.path.join(intermediate_iteration_folder,
                                                                f"ep{epoch + 1}_iter{total_iterations + 1}.png")
                    save_image(intermediate_image, intermediate_iteration_path, format='png')

            except Exception as e:
                print(f"Iteration {total_iterations + 1} failed with error: {e}")
                print(f"Full traceback: {e}", flush=True)

            total_loss.backward()
            optimizer_gen.step()

            total_iterations += 1

        
        save_model(generator, discriminator, epoch, output_path)

        
        writer.add_scalar("Total Loss", total_loss.item(), epoch)

    print("Training complete.")
    writer.close()


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


if __name__ == "__main__":
    main()
