import os
import logging
import torch
import numpy as np
from torchvision import transforms
from PIL import Image, ImageOps, ImageDraw
from scipy.ndimage import binary_fill_holes
from skimage.filters import threshold_otsu, threshold_local
import matplotlib.pyplot as plt
from skimage import morphology, filters, measure
import cv2

#
# Special note: no NVIDIA supprot here, relies on mps as-is
#
def mask_background(image, block_size=35, min_size=5000):
    """
    Mask out the background of the image by setting it to black.

    Parameters:
    image (np.ndarray): Input brain image (skull_stripped already).
    block_size (int): Block size for adaptive thresholding.
    min_size (int): Minimum size for small object removal.

    Returns:
    np.ndarray: Image with background masked out.
    """
    # Adaptive thresholding
    adaptive_thresh = threshold_local(image, block_size, offset=10)
    binary_mask = image > adaptive_thresh
    # Remove small objects
    cleaned_mask = morphology.remove_small_objects(binary_mask, min_size=min_size)

    # Fill holes within objects
    filled_mask = binary_fill_holes(cleaned_mask)

    # Label connected components
    labeled_mask = measure.label(filled_mask)

    # Keep the largest connected component which should be the brain region
    props = measure.regionprops(labeled_mask)
    if len(props) > 0:
        largest_region = max(props, key=lambda x: x.area)
        brain_mask = labeled_mask == largest_region.label
        image[~brain_mask] = 0  # Set background to black

    props = measure.regionprops(labeled_mask)

    return labeled_mask

    # Visualize intermediate steps
    #    plt.figure(figsize=(12, 8))
    #    plt.subplot(1, 3, 1)
    #    plt.imshow(binary_mask, cmap='gray')
    #    plt.title('Binary Mask')
    #    plt.subplot(1, 3, 2)
    #    plt.imshow(filled_mask, cmap='gray')
    #    plt.title('Filled Mask')
    #    plt.subplot(1, 3, 3)
    #    plt.imshow(image, cmap='gray')
    #    plt.title('Masked Image')
    #plt.show()

    return image



def largest_connected_component(binary_mask):
    labeled_mask, num_labels = measure.label(binary_mask, connectivity=2, return_num=True)
    if num_labels == 0:
        logging.info("Didn't find largest area")
        return binary_mask
    largest_label = 1 + np.argmax(np.bincount(labeled_mask.flat)[1:])
    return labeled_mask == largest_label

def load_image(image_path):
    """Load an image from a file path and convert to grayscale numpy array."""
    try:
        image = Image.open(image_path).convert('L')
        logging.info(f"Image loaded: {image.size}")
        return np.array(image) / 255.0
    except Exception as e:
        logging.error(f"Error loading image: {e}")
        raise

def preprocess_image(image):
    try:
        logging.info("Starting image preprocessing")
        # Normalization
        image = cv2.normalize(image, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
        logging.info("Image normalization completed")
        return image
    except Exception as e:
        logging.error(f"Error during image preprocessing: {e}")
        raise

def skull_stripping(image, output_path="/tmp/", min_size=1000, area_threshold=5000):
    """
    Perform skull stripping on the brain MRI image.

    Parameters:
    image (np.ndarray): Input brain image.
    output_path (str): Path to save the visual representation of skull stripping.
    min_size (int): Minimum size for small object removal.
    area_threshold (int): Area threshold for small hole removal.

    Returns:
    tuple: Skull stripped image and brain mask.
    """
    try:
        logging.info("Starting skull stripping")
        _, binary_mask = cv2.threshold((image * 255).astype(np.uint8), 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        binary_mask = morphology.remove_small_objects(binary_mask > 0, min_size=min_size)
        binary_mask = morphology.remove_small_holes(binary_mask, area_threshold=area_threshold)
        brain_mask = largest_connected_component(binary_mask)
        brain_extracted = image * brain_mask

        # Save visual representation of skull stripping
        plt.figure(figsize=(10, 5))
        plt.subplot(1, 2, 1)
        plt.title('Original Image')
        plt.imshow(image, cmap='gray')

        plt.subplot(1, 2, 2)
        plt.title('Skull Stripped Image')
        plt.imshow(brain_extracted, cmap='gray')

        plt.savefig(output_path + "_skull_stripping.png")
        plt.close()

        logging.info("Skull stripping completed")
        return brain_extracted, brain_mask
    except Exception as e:
        logging.error(f"Error during skull stripping: {e}")
        raise

def apply_red_mask(image, mask):
    """Apply a red mask to the identified areas in the image."""
    red_mask = Image.new('RGBA', image.size, (255, 0, 0, 0))  # Create a transparent image
    draw = ImageDraw.Draw(red_mask)
    width, height = image.size

    for x in range(width):
        for y in range(height):
            if mask.getpixel((x, y)) > 0:  # Check if the mask pixel is identified (non-zero)
                draw.point((x, y), (255, 0, 0, 128))  # Red with transparency
    return Image.alpha_composite(image.convert('RGBA'), red_mask)


def process_image(image_path, model, preprocess): #This is just awful but was an early attempt
    logging.info(f"Loading image from {image_path}")
    image = Image.open(image_path).convert('L')  # Ensure the image is in grayscale
    image = np.array(image)
    logging.info(f"Original image shape: {image.shape}, min: {np.min(image)}, max: {np.max(image)}")

    # Preprocess the image
    #image_pr = preprocess_image(image.copy())
    #logging.info(f"After preprocessing: shape: {image_pr.shape}, min: {np.min(image_pr)}, max: {np.max(image_pr)}")

    # Apply skull stripping
    x, image_np  = skull_stripping(image.copy())
    logging.info(f"After skull stripping: shape: {image_np.shape}, min: {np.min(image_np)}, max: {np.max(image_np)}")

    if image_np is None:
        raise ValueError("Skull stripping returned None")

    # Mask the background
    image_np2 = mask_background(image_np.copy())
    logging.info(f"After background masking: shape: {image_np2.shape}, min: {np.min(image_np2)}, max: {np.max(image_np2)}")

    if image_np2 is None:
        raise ValueError("Background masking returned None")

    # Convert back to PIL image for resizing and further processing
    image_fixed = Image.fromarray(image_np2.copy())
    original_size = image_fixed.size
    logging.info(f"Original image size: {original_size}")

    # Resize the image to 256x256
    image_npn = image_fixed.resize((256, 256))
    logging.info(f"Resized image size: {image_npn.size}")

    # Convert the single-channel image to a 3-channel image
    #image_3ch = np.stack([np.array(image_npn)] * 3, axis=-1)
    image_3ch = np.stack([np.array(image_np2)] * 3, axis=-1)
    logging.info(f"3-channel image shape: {image_3ch.shape}")

    # Apply z-score normalization per volume
    image_normalized = (image_3ch - np.mean(image_3ch)) / np.std(image_3ch)
    logging.info(f"Image normalization completed with shape: {image_normalized.shape}, mean: {np.mean(image_normalized)}, std: {np.std(image_normalized)}")

    # Apply transforms
    input_tensor = preprocess(image_normalized).float()  # Ensure the tensor is of type Float
    input_batch = input_tensor.unsqueeze(0)

    # Move to GPU if available
    if torch.backends.mps.is_available():
        input_batch = input_batch.to('mps')
        model = model.to('mps')

    # Debug prints to check the dimensions
    print(f"Input batch shape: {input_batch.shape}")

    # Make prediction
    with torch.no_grad():
        output = model(input_batch)

    # Post-process the output
    output_image = torch.round(output[0]).cpu().numpy().squeeze()
    output_image = (output_image * 255).astype(np.uint8)  # Scale to 0-255
    output_image = mask_background(output_image.copy())
    # Convert output to PIL image
    output_image = Image.fromarray(output_image).resize(original_size)

    # Superimpose the mask onto the original image
    original_image = image_npn # Image.open(image_path).convert('L')  # Load original image again
    superimposed_image = Image.blend(original_image.convert('RGB'), output_image.convert('RGB'), alpha=0.3)

    # Create a side-by-side comparison image
    original_image_rgb = original_image.convert('RGB')
    comparison_image = Image.new('RGB', (original_image_rgb.width * 3, original_image_rgb.height))
    comparison_image.paste(original_image_rgb, (0, 0))
    comparison_image.paste(superimposed_image, (original_image_rgb.width, 0))

    superimposed_image = apply_red_mask(image_npn, output_image)

    # Create a side-by-side comparison image
    original_image_rgb = original_image.convert('RGB')
    comparison_image = Image.new('RGB', (original_image_rgb.width * 2, original_image_rgb.height))
    comparison_image.paste(original_image_rgb, (0, 0))
    comparison_image.paste(superimposed_image.convert('RGB'), (original_image_rgb.width, 0))

    return comparison_image


def main(input_dir, output_dir):
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    model = torch.hub.load('mateuszbuda/brain-segmentation-pytorch', 'unet',
                           in_channels=3, out_channels=1, init_features=32, pretrained=True)

    # Define the preprocess transforms
    preprocess = transforms.Compose([
        transforms.ToTensor(),
    ])

    # Process each image in the input directory and collect comparison images
    comparison_images = []
    for filename in os.listdir(input_dir):
        if filename.endswith(('.png', '.jpg', '.jpeg')):
            image_path = os.path.join(input_dir, filename)
            comparison_image = process_image(image_path, model, preprocess)
            comparison_images.append(comparison_image)
            output_filename = os.path.join(output_dir, f"comparison_{filename}")
            comparison_image.save(output_filename)
            print(f"Processed and saved: {output_filename}")

    # Combine all comparison images into one
    combined_width = max(img.width for img in comparison_images)
    combined_height = sum(img.height for img in comparison_images)
    combined_image = Image.new('RGB', (combined_width, combined_height))

    y_offset = 0
    for img in comparison_images:
        combined_image.paste(img, (0, y_offset))
        y_offset += img.height

    combined_image_path = os.path.join(output_dir, "combined_comparison.png")
    combined_image.save(combined_image_path)
    print(f"Combined image saved: {combined_image_path}")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    input_directory = "input_files/"
    output_directory = "output_files/"
    main(input_directory, output_directory)
