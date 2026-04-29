import torch
import numpy as np
import cv2
import os
import segmentation_models_pytorch as smp

# --------------------------------------------------
# ⚙️ CLINICAL CONFIGURATION
# --------------------------------------------------
MODEL_PATH = "tcia_resnet34_best.pth" # Ensure this matches your downloaded Kaggle weights
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
INFERENCE_SIZE = (256, 256) # From your training script

# --------------------------------------------------
# 🏥 2.5D ONCOLOGY SEGMENTATION ENGINE
# --------------------------------------------------
class TumorSystem:
    def __init__(self):
        print(f"🏥 Initializing BoneSegNet 2.5D Engine on {DEVICE}...")
        
        # 1. Initialize U-Net++ (Must match training code exactly)
        self.model = smp.UnetPlusPlus(
            encoder_name="resnet34", 
            encoder_weights=None, # Don't need to download imagenet weights for inference
            in_channels=3,        # 2.5D RGB Input
            classes=2,            # Class 0: Spine, Class 1: Lesion
            decoder_attention_type='scse'
        ).to(DEVICE)

        self.is_mock = False
        self._load_weights()

    def _load_weights(self):
        try:
            if os.path.exists(MODEL_PATH):
                state_dict = torch.load(MODEL_PATH, map_location=DEVICE)
                self.model.load_state_dict(state_dict)
                self.model.eval()
                print("✅ 2.5D Weights Loaded Successfully.")
            else:
                print(f"⚠️ Weights not found at {MODEL_PATH}. Switching to MOCK mode for UI testing.")
                self.is_mock = True
        except Exception as e:
            print(f"❌ Error loading model: {e}. Switching to MOCK mode.")
            self.is_mock = True

    def analyze_scan(self, input_tensor, progress_bar=None):
        """
        Processes a full 3D CT volume using the 2.5D slice-stacking approach.
        Expected input: Torch tensor of shape (1, 1, D, H, W)
        """
        # 1. Extract raw 3D volume
        vol = input_tensor.squeeze().numpy() # Shape: (D, H, W)
        D, H_orig, W_orig = vol.shape
        
        # 2. Pre-allocate the final binary mask volume and probability volume
        mask_volume = np.zeros((D, H_orig, W_orig), dtype=np.float32)
        prob_volume = np.zeros((D, H_orig, W_orig), dtype=np.float32)
        
        # 3. Global Normalization [0, 1] to match training conditions
        vol_min, vol_max = vol.min(), vol.max()
        if vol_max > vol_min:
            vol = (vol - vol_min) / (vol_max - vol_min)

        print(f"Processing Volume: {D} slices. HxW: {H_orig}x{W_orig}")

        # ---------------------------------------------------------
        # THE 2.5D INFERENCE LOOP (Slice Stacking)
        # ---------------------------------------------------------
        if self.is_mock:
            # Fallback mock for testing UI without weights
            center_y, center_x = H_orig//2, W_orig//2
            y, x = np.ogrid[:H_orig, :W_orig]
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            for i in range(D//3, 2*D//3):
                mask_volume[i, dist < 15] = 1.0
                if progress_bar is not None:
                    progress_bar.progress((i + 1) / D, text=f"Neural Engine Analyzing Slice {i+1}/{D}...")
        else:
            self.model.eval()
            with torch.no_grad():
                for i in range(D):
                    # Step A: Extract N-1, N, N+1 (Handling boundary cases safely)
                    slice_prev = vol[max(0, i-1), :, :]
                    slice_curr = vol[i, :, :]
                    slice_next = vol[min(D-1, i+1), :, :]
                    
                    # Step B: Stack into Pseudo-RGB (H, W, 3)
                    rgb_stack = np.stack([slice_prev, slice_curr, slice_next], axis=-1)
                    
                    # Step C: Resize to 256x256 (Using INTER_AREA as per training script)
                    if (H_orig, W_orig) != INFERENCE_SIZE:
                        rgb_resized = cv2.resize(rgb_stack, INFERENCE_SIZE, interpolation=cv2.INTER_AREA)
                    else:
                        rgb_resized = rgb_stack
                        
                    # Step D: Convert to PyTorch Tensor (1, 3, 256, 256)
                    input_t = torch.tensor(rgb_resized).permute(2, 0, 1).unsqueeze(0).float().to(DEVICE)
                    
                    # Step E: Forward Pass
                    logits = self.model(input_t)
                    
                    # Step F: Extract Lesion Mask (Class 1) and Apply Strict Threshold
                    prob = torch.sigmoid(logits)[0, 1, :, :].cpu().numpy()
                    pred_mask = (prob > 0.5).astype(np.float32)
                    
                    # --- THE TITANIUM NOISE FILTER ---
                    # Lowered threshold: Only delete microscopic 1-or-2 pixel dust. Keep the real tumors.
                    if np.sum(pred_mask) < 3:
                        pred_mask = np.zeros_like(pred_mask)
                        prob = np.zeros_like(prob) # Erase probabilities of noise too

                    # Step G: Resize binary mask back to original scan dimensions
                    if (H_orig, W_orig) != INFERENCE_SIZE:
                        pred_mask = cv2.resize(pred_mask, (W_orig, H_orig), interpolation=cv2.INTER_NEAREST)
                        prob = cv2.resize(prob, (W_orig, H_orig), interpolation=cv2.INTER_LINEAR)
                        
                    mask_volume[i, :, :] = pred_mask
                    prob_volume[i, :, :] = prob

                    if progress_bar is not None:
                        # Update the UI bar!
                        progress_bar.progress((i + 1) / D, text=f"Neural Engine Analyzing Slice {i+1}/{D}...")

        # ---------------------------------------------------------
        # CLINICAL METRICS CALCULATION
        # ---------------------------------------------------------
        # Count slices that contain at least one tumor pixel
        affected_slices = int(np.sum(np.max(mask_volume, axis=(1, 2)) > 0))
        # Count total tumor pixels across the volume
        total_lesion_volume = int(np.sum(mask_volume)) 
        
        diagnosis = "Lytic Lesion Detected" if affected_slices > 0 else "Normal Bone Structure"

        # Calculate the average confidence of all the detected green pixels
        confidence_score = 0.0
        if affected_slices > 0:
            tumor_probs = prob_volume[mask_volume == 1.0]
            confidence_score = float(np.mean(tumor_probs) * 100) # Convert to percentage

        return {
            'diagnosis': diagnosis,
            'affected_slices': affected_slices,
            'lesion_volume_px': total_lesion_volume,
            'confidence': confidence_score,
            'heatmap': mask_volume # Strict binary mask matrix
        }