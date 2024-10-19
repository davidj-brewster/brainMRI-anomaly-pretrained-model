import torch
import numpy as np
from PIL import Image
from torchvision import transforms

def preprocess_normalise_image(input_image):
    # Ensure the image is in grayscale mode ('L') if not already
    if input_image.mode != 'L':
        input_image = input_image.convert('L')

    # Convert image to numpy array
    image = np.array(input_image.copy())
    image = np.array(image) / 255.0
    # Calculate mean and std for normalization based on the single channel
    m, s = np.mean(image), np.std(image)

    # Define the preprocess transforms without stacking yet
    preprocess = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[m], std=[s]),  # Normalize as a single-channel image
    ])

    # Apply the transformations
    input_tensor = preprocess(image.copy()).float()

    # Stack the single channel tensor into 3 channels for model compatibility
    input_tensor = input_tensor.repeat(3, 1, 1)

    # Create a batch dimension
    input_batch = input_tensor.unsqueeze(0)

    return input_batch


def main():
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                            in_channels=3, out_channels=1, init_features=32, pretrained=True)

    input_file_name = "brain9.png"
    input_directory = "./input_files"  # Add to .gitignore to exclude from version control

    try:
        # Load the image
        input_image = Image.open(f"{input_directory}/{input_file_name}")
    except FileNotFoundError as e:
        print(f"File Not Found: {input_directory}/{input_file_name} - full exception is: {e}")
        raise e

    # Preprocess the image
    input_batch = preprocess_normalise_image(input_image)

    # Check and move to appropriate device
    device = torch.device('cuda' if torch.cuda.is_available() else 'mps' if torch.backends.mps.is_available() else 'cpu')
    model = model.to(device)
    input_batch = input_batch.to(device)

    # Example: Perform model inference
    with torch.no_grad():
        output = model(input_batch)

    print("Inference completed successfully.")


if __name__ == "__main__":
    main()

