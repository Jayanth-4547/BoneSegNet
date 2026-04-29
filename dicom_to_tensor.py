import os
import cv2
import torch
import pydicom
import numpy as np

# 1. Point this to the folder containing Patient 1's .dcm files
dicom_dir = "./MonoE_40keVHU" 

print(f"Loading raw hospital DICOM files from {dicom_dir}...")

# 2. Load all DICOM files
dicom_files = [os.path.join(dicom_dir, f) for f in os.listdir(dicom_dir) if f.endswith('.dcm')]
slices = [pydicom.dcmread(f) for f in dicom_files]

# 3. CRITICAL: Sort slices by physical Z-axis location so the 3D stack is in the correct order
slices.sort(key=lambda x: float(x.ImagePositionPatient[2]))

volume = []

print("Applying Hounsfield Windowing and standardizing dimensions...")
for s in slices:
    # A. Extract raw pixel data
    image = s.pixel_array.astype(np.float32)
    
    # B. Convert to Hounsfield Units (HU) using DICOM metadata
    intercept = float(s.RescaleIntercept)
    slope = float(s.RescaleSlope)
    hu_image = image * slope + intercept
    
    # C. Apply the Strict Bone Window (-200 to +1500) defined in your report
    hu_image = np.clip(hu_image, -200, 1500)
    
    # D. Normalize to [0, 1] for the neural network
    normalized = (hu_image - (-200)) / (1500 - (-200))
    
    # E. Resize to 256x256 (The exact resolution your 2.5D ResNet34 expects)
    resized = cv2.resize(normalized, (256, 256), interpolation=cv2.INTER_AREA)
    
    volume.append(resized)

# 4. Stack into a 3D numpy array (Depth, Height, Width)
volume_array = np.stack(volume, axis=0)

# 5. Convert to PyTorch Tensor format: (Batch, Channel, Depth, Height, Width)
tensor_scan = torch.tensor(volume_array).unsqueeze(0).unsqueeze(0)

# 6. Save the final file for your Web UI
save_name = "real_demo_patient_1.pt"
torch.save(tensor_scan, save_name)

print(f"✅ Success! Saved {save_name} with shape {tensor_scan.shape}")
print("Your raw hospital data is now perfectly formatted for the BoneSegNet UI.")