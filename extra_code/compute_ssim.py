import os
from skimage.metrics import structural_similarity as ssim
import cv2
import numpy as np

# Define the directories
codec='prores'
raw_data_tiff_dir = '/mnt/altnas/work/dimos/raw_data_tiff'
data_dir = f'/mnt/altnas/work/dimos/SPARZ_fig1/no_ROI/{codec}/images'

def compute_average_ssim(raw_data_tiff_dir, data_dir):
    ssim_scores = []

    # List all TIFF files in the raw_data_tiff folder
    raw_tiff_files = [f for f in os.listdir(raw_data_tiff_dir) if f.endswith('.tiff')]

    for raw_file in raw_tiff_files:
        raw_file_name = os.path.splitext(raw_file)[0]
        matched_file = None

        # Search for matching file in the data directory
        for data_file in os.listdir(data_dir):
            if raw_file_name in os.path.splitext(data_file)[0]:
                matched_file = data_file
                break

        if matched_file:
            # Load the images
            raw_image_path = os.path.join(raw_data_tiff_dir, raw_file)
            matched_image_path = os.path.join(data_dir, matched_file)

            raw_image = cv2.imread(raw_image_path, cv2.IMREAD_GRAYSCALE)
            matched_image = cv2.imread(matched_image_path, cv2.IMREAD_GRAYSCALE)

            # Ensure the images are the same size
            height, width = raw_image.shape
            matched_image = cv2.resize(matched_image, (width, height))

            # Compute SSIM
            score = ssim(raw_image, matched_image)
            ssim_scores.append(score)

    # Compute the average SSIM
    if ssim_scores:
        average_ssim = np.mean(ssim_scores)
    else:
        average_ssim = None

    return average_ssim

average_ssim = compute_average_ssim(raw_data_tiff_dir, data_dir)
print(f"{codec} Average SSIM: {average_ssim}")
