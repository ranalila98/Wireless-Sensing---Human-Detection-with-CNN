import os
import numpy as np
import matplotlib.pyplot as plt

# Load datasets
dataset_no_presence = np.load('dataset/dataset-no-presence.npy')
dataset_presence = np.load('dataset/dataset-presence.npy')

print("Dataset (No Presence) shape:", dataset_no_presence.shape)
print("Dataset (Presence) shape:", dataset_presence.shape)

# output directories 
output_dir_presence = "generated_images/Human_Presence"
output_dir_no_presence = "generated_images/No_Presence"

# Create directories if they don't exist
os.makedirs(output_dir_presence, exist_ok=True)
os.makedirs(output_dir_no_presence, exist_ok=True)

# Function to compute the periodogram using 2D-FFT
def compute_periodogram(csi_data):
    periodogram = np.abs(np.fft.fft2(csi_data))  # Compute 2D FFT magnitude
    periodogram = np.fft.fftshift(periodogram)  # Center the zero frequency component
    periodogram = (periodogram - np.min(periodogram)) / (np.max(periodogram) - np.min(periodogram))  # Normalize to [0,1]
    return periodogram

for capture_index in range(250):
    print(f"Processing Capture Index: {capture_index}")

    # Compute periodograms for presence and no-presence captures
    periodogram_no_presence = compute_periodogram(dataset_no_presence[:, :, capture_index])
    periodogram_presence = compute_periodogram(dataset_presence[:, :, capture_index])

    # Function to plot and save the periodogram
    def save_periodogram(data, title, save_path):
        fig, ax = plt.subplots(figsize=(6, 5))  # Create single plot
        c = ax.imshow(data, cmap='jet', aspect='auto', origin='lower', interpolation='sinc')
        fig.colorbar(c, ax=ax, label="Power Spectrum Intensity")
        ax.set_xlabel("Temporal Index (Packets)")
        ax.set_ylabel("Spatial Index (Subcarriers)")
        ax.set_title(title)
        
        plt.savefig(save_path, dpi=300, bbox_inches='tight')  
        plt.close(fig)  

  
    save_path_no_presence = os.path.join(output_dir_no_presence, f"capture_{capture_index:03d}.png")
    save_path_presence = os.path.join(output_dir_presence, f"capture_{capture_index:03d}.png")

    save_periodogram(periodogram_no_presence, f"CSI Periodogram - No Presence (Capture {capture_index})", save_path_no_presence)
    save_periodogram(periodogram_presence, f"CSI Periodogram - Presence (Capture {capture_index})", save_path_presence)

print("All images saved successfully in respective folders")
