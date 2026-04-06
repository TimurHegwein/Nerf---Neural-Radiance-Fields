import os
import shutil
import matplotlib.pyplot as plt
from nilearn import datasets, plotting

def download_and_preview_brain(output_dir="brains"):
    """
    Downloads the ICBM 152 T1 template and displays a 3-axis preview.
    """
    os.makedirs(output_dir, exist_ok=True)
    target_path = os.path.join(output_dir, "brain_0.nii.gz")

    # 1. Check if file already exists to save bandwidth
    if os.path.exists(target_path):
        print(f"File already exists at {target_path}. Skipping download.")
    else:
        print("Fetching ICBM 152 T1 Template (High-Res 2009 version)...")
        data = datasets.fetch_icbm152_2009()
        source_path = data['t1']
        shutil.copy(source_path, target_path)
        print(f"Success! Brain saved to: {target_path}")

    # 2. Visualize the downloaded brain
    print("Generating preview...")
    # 'plot_anat' is the standard for structural (anatomical) images
    plotting.plot_anat(
        target_path, 
        title="ICBM 152 T1 - Downloaded Template", 
        display_mode='ortho', # Shows Sagittal, Coronal, and Axial views
        colorbar=True,
        annotate=True
    )
    
    print("Opening preview window. Close it to proceed.")
    plotting.show()

if __name__ == "__main__":
    download_and_preview_brain()