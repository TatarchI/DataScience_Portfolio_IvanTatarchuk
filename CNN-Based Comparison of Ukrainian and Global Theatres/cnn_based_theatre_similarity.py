# --------------------------- 🔹 Theatre Similarity System  ------------------------------------

"""
Author: Ivan Tatarchuk

Project Description:
--------------------
This script implements a visual similarity system for comparing the facades of world-famous theaters.
The main goal is to semantically and structurally compare Ukrainian theaters (Kyiv, Odesa, Lviv)
to 100 internationally renowned theaters using deep learning techniques.

The architecture leverages pretrained convolutional neural networks (CNNs), particularly ResNet50,
to extract global visual descriptors from processed images. Cosine similarity is used to measure
the resemblance between the Ukrainian theater and each item in the global dataset.

Additionally, Grad-CAM visualization is applied to highlight the regions of the image
that contribute most to the model’s understanding — ensuring transparency and explainability.

🔹 Use Cases:
- Architectural analysis & urban heritage research
- Visual localization & cross-country landmark identification
- Educational computer vision demonstrations for CNN interpretability

🔹 Key Techniques:
- Pretrained ResNet50 for feature extraction
- Cosine similarity for ranking visual similarity
- Grad-CAM for CNN attention visualization
- Image preprocessing with resizing, contrast enhancement, and noise reduction

Package Dependencies:
---------------------
pip                          25.0.1
numpy                        1.26.4
matplotlib                   3.10.1
cv2                          4.11.0.86
torch                        2.4.1
torchvision                  0.19.1
torchcam                     0.4.0
transformers                 4.46.3
"""

# --------------------------- 🔹 Model Initialization & Configuration ------------------------------------

# Core libraries
import torch
import torchvision.transforms as transforms
from torch.nn.functional import cosine_similarity

# Grad-CAM for model explainability
from torchcam.methods import GradCAM

# Pretrained convolutional model (ResNet50)
from torchvision.models import resnet50

# Image and visualization utilities
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Suppress warnings for cleaner output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# 🔹 Select computation device — GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# 🔹 Initialize ResNet50 for image descriptor extraction
# We use a pretrained ResNet50 model from torchvision, with the classification head removed.
# This allows us to use the 2048-dimensional output of the global average pooling layer as a visual descriptor.
cnn_model = resnet50(pretrained=True)
cnn_model.fc = torch.nn.Identity()  # Remove classification layer
cnn_model = cnn_model.to(device).eval()  # Set to evaluation mode


# 🔹 Initialize a second copy of ResNet50 for Grad-CAM visualization
# Grad-CAM requires gradient computation — so we keep this model in train mode.
cnn_model_cam = resnet50(pretrained=True)
cnn_model_cam.fc = torch.nn.Identity()
cnn_model_cam = cnn_model_cam.to(device).train()  # Required for proper CAM behavior

# Extract Grad-CAM activations from the final convolutional block (layer4)
cam_extractor = GradCAM(cnn_model_cam, target_layer="layer4")

# --------------------------- 🔹 Image Loading Utility ------------------------------------

def load_image(path: str, show: bool = False) -> np.ndarray:
    """
    Loads an image from a given file path using OpenCV.
    Optionally visualizes it with matplotlib if `show=True`.

    Parameters:
        path (str): Path to the image file.
        show (bool): If True, displays the image using matplotlib.

    Returns:
        np.ndarray: Loaded image in OpenCV format (BGR). Returns None if loading fails.
    """
    image = cv2.imread(path)

    if show:
        # Convert BGR (OpenCV) to RGB (matplotlib) before visualization
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        plt.figure(figsize=(6, 6))
        plt.imshow(image_rgb)
        plt.title(f"Image: {os.path.basename(path)}")
        plt.axis("off")
        plt.show()

    return image

# --------------------------- 🔹 Image Size Report Generator ------------------------------------

def collect_image_sizes(root_dirs, output_file="image_resolution_report.txt"):
    """
    Scans all image files within the specified directories and collects their resolutions.
    Generates a report file with detailed size statistics for each image.

    Parameters:
        root_dirs (list): List of directory names to scan (e.g., ["Ukraine", "theatre_dataset"]).
        output_file (str): Filename for the generated text report.

    Returns:
        List[Tuple[str, int, int]]: A sorted list of (filename, width, height) tuples.
    """
    sizes = []

    for root_dir in root_dirs:
        for filename in os.listdir(root_dir):
            if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
                continue  # Skip non-image files

            path = os.path.join(root_dir, filename)
            img = cv2.imread(path)
            if img is not None:
                h, w = img.shape[:2]
                sizes.append((filename, w, h))
            else:
                print(f"⚠️ Unable to read file: {filename}")

    # Sort images by area (width * height)
    sizes.sort(key=lambda x: x[1] * x[2])

    # Write to report file
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(f"📊 Image Size Report ({len(sizes)} images)\n")
        f.write("=" * 50 + "\n")
        for name, w, h in sizes:
            line = f"{name:50} — {w:4} x {h:4} = {w*h:8,d} px\n"
            f.write(line)

    print(f"\n✅ Report successfully saved to file: {output_file}")
    return sizes

# --------------------------- 🔹 Image Preprocessing for Descriptor Extraction ------------------------------------

def preprocess_image_for_matching(image: np.ndarray) -> np.ndarray:
    """
    Preprocesses an image to standardize its size and visual quality before feature extraction.

    Key operations:
    - Resize to fixed resolution (600x400) to ensure uniform input dimensions.
    - Apply CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance local contrast.
    - Apply Gaussian blur to reduce noise and small details, which may disrupt global descriptor extraction.

    This function ensures consistent input format for both ResNet and CLIP models.

    Parameters:
        image (np.ndarray): Input image in BGR format (as read by OpenCV).

    Returns:
        np.ndarray: Preprocessed image ready for embedding.
    """
    target_w, target_h = 600, 400

    # 🔹 Resize to fixed resolution
    image = cv2.resize(image, (target_w, target_h), interpolation=cv2.INTER_AREA)

    # 🔹 Convert to LAB color space and enhance contrast using CLAHE
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    l_clahe = clahe.apply(l)
    lab = cv2.merge((l_clahe, a, b))
    enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)

    # 🔹 Apply Gaussian blur for noise reduction
    final = cv2.GaussianBlur(enhanced, (3, 3), 0)

    return final

# --------------------------- 🔹 Global Descriptor Extraction via ResNet50 ------------------------------------

def extract_resnet_descriptor(image: np.ndarray, use_gradcam: bool = False) -> torch.Tensor:
    """
    Extracts a global image descriptor using a pretrained ResNet50 model.

    This function transforms an input image into a 2048-dimensional feature vector
    using the ResNet50 backbone (with the classification head removed).
    These descriptors are suitable for computing global similarity via cosine distance.

    Parameters:
        image (np.ndarray): Input image in BGR format (as loaded via OpenCV).
        use_gradcam (bool): If True, enables gradient tracking for Grad-CAM visualization.
                            Otherwise, disables gradients for inference.

    Returns:
        torch.Tensor: 2048-dimensional global feature vector.
    """
    # Define preprocessing pipeline: resize, normalize (as expected by ResNet)
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],  # ImageNet mean
            std=[0.229, 0.224, 0.225]    # ImageNet std
        )
    ])

    # Convert from BGR (OpenCV) to RGB (PyTorch convention)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    input_tensor = transform(rgb_image).unsqueeze(0).to(device)

    # Enable gradient tracking if needed (for Grad-CAM)
    if use_gradcam:
        input_tensor.requires_grad_()
        descriptor = cnn_model(input_tensor)
    else:
        with torch.no_grad():
            descriptor = cnn_model(input_tensor)

    return descriptor.squeeze(0)

# --------------------------- 🔹 Grad-CAM Visualization for Image Pairs ------------------------------------

def visualize_gradcam_pair(
    image1: np.ndarray,
    image2: np.ndarray,
    model: torch.nn.Module,
    extractor: GradCAM,
    title1: str,
    title2: str
):
    """
    Visualizes Grad-CAM heatmaps for a pair of images to highlight which regions
    the CNN model considers most important for its decision.

    This function applies Grad-CAM to each image independently and shows a side-by-side
    comparison of their attention heatmaps, which can be useful for interpreting
    model focus areas and verifying similarity rationale.

    Parameters:
        image1 (np.ndarray): First input image in BGR format.
        image2 (np.ndarray): Second input image in BGR format.
        model (torch.nn.Module): The CNN model used for inference (set to training mode).
        extractor (GradCAM): Initialized GradCAM object attached to a model layer.
        title1 (str): Title label for the first image (e.g. "Kyiv Opera").
        title2 (str): Title label for the second image (e.g. "Teatro alla Scala").
    """

    # Define preprocessing consistent with ResNet input expectations
    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    def get_cam_overlay(image: np.ndarray) -> np.ndarray:
        """
        Generates a Grad-CAM heatmap overlay for a single image.

        Returns:
            np.ndarray: BGR image with heatmap overlaid.
        """
        input_tensor = transform(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(device)

        # Reset gradients and run model forward
        model.zero_grad()
        output = model(input_tensor)
        class_idx = output[0].argmax().item()

        # Generate CAM activation map from the specified layer
        cam_map = extractor(class_idx, output)[0].squeeze().cpu().numpy()
        cam_map = cv2.resize(cam_map, (image.shape[1], image.shape[0]))
        cam_map = np.clip(cam_map, 0, 1)

        # Convert to uint8 format and generate colored heatmap
        cam_map_uint8 = (cam_map * 255).astype(np.uint8)
        if cam_map_uint8.ndim != 2:
            cam_map_uint8 = cam_map_uint8.squeeze()
        heatmap = cv2.applyColorMap(cam_map_uint8, cv2.COLORMAP_JET)

        # Blend heatmap with original image
        overlay = cv2.addWeighted(image, 0.6, heatmap, 0.4, 0)
        return overlay

    # Compute Grad-CAM overlays for both images
    cam1 = get_cam_overlay(image1)
    cam2 = get_cam_overlay(image2)

    # Concatenate and display side-by-side in RGB format
    combined = np.hstack((cam1, cam2))
    combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

    plt.figure(figsize=(14, 5))
    plt.imshow(combined_rgb)
    plt.title(f"Grad-CAM: {title1} vs {title2}")
    plt.axis("off")
    plt.tight_layout()
    plt.show()

# --------------------------- 🔹 MAIN SCRIPT: Similarity Search & Grad-CAM Explanation ---------------------------

if __name__ == "__main__":
    # 🔍 Step 1: Collect image resolutions for Ukrainian and world theatre datasets
    image_sizes = collect_image_sizes(["Ukraine", "theatre_dataset"])

    # 🎭 Step 2: Select the Ukrainian reference theatre for comparison
    # Options: "Kyiv", "Odesa", "Lviv"
    MODE = "Odesa"
    base_path = f"Ukraine/{MODE}_Opera_Theatre_{MODE}.jpg"

    # 📥 Step 3: Load and preprocess the reference image (Ukrainian theatre)
    img_base = load_image(base_path, show=True)
    if img_base is None:
        raise ValueError(f"❌ Failed to load base image: {base_path}")
    img_base = preprocess_image_for_matching(img_base)
    descriptor_base = extract_resnet_descriptor(img_base, use_gradcam=False)

    # 🌍 Step 4: Loop through all world theatre images and compute similarity
    theatre_dir = "theatre_dataset"
    similarities = []

    for filename in os.listdir(theatre_dir):
        if not filename.lower().endswith((".jpg", ".jpeg", ".png")):
            continue  # Skip unsupported files

        path = os.path.join(theatre_dir, filename)
        img = load_image(path, show=False)
        if img is None:
            print(f"⚠️ Skipped: {filename} (could not be read)")
            continue

        img = preprocess_image_for_matching(img)
        descriptor = extract_resnet_descriptor(img)

        # 📐 Cosine similarity between the Ukrainian theatre and current world theatre
        similarity = cosine_similarity(descriptor.unsqueeze(0), descriptor_base.unsqueeze(0)).item()
        similarities.append((filename, similarity))

    # 📊 Step 5: Sort by cosine similarity (descending) and get top 5 matches
    similarities.sort(key=lambda x: x[1], reverse=True)
    top_matches = similarities[:5]

    print("\n🔝 Top 5 most similar theatres based on ResNet50 cosine similarity:")
    for i, (name, score) in enumerate(top_matches, 1):
        print(f"{i}. {name} — similarity: {score:.4f}")

    # 🖼️ Step 6: Visualize side-by-side comparison for top-5 similar theatres
    print("\n📸 Visualizing top-5 theatre matches:")
    for i, (name, score) in enumerate(top_matches, 1):
        matched_path = os.path.join(theatre_dir, name)
        matched_img = load_image(matched_path, show=False)
        matched_img = preprocess_image_for_matching(matched_img)

        combined = np.hstack((img_base, matched_img))
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 4))
        plt.imshow(combined_rgb)
        plt.title(f"{i}. {name} — similarity: {score:.4f}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

    # 🔬 Step 7: Visual Grad-CAM attention comparison for each match
    print("\n🌡️ Grad-CAM attention analysis for top matches:")
    for i, (name, score) in enumerate(top_matches, 1):
        matched_path = os.path.join(theatre_dir, name)
        matched_img = load_image(matched_path, show=False)
        matched_img = preprocess_image_for_matching(matched_img)

        print(f"\n🌡️ Grad-CAM Comparison #{i}: {MODE}_Opera_Theatre vs {name}")
        visualize_gradcam_pair(
            img_base, matched_img,
            cnn_model_cam, cam_extractor,
            f"{MODE}_Opera_Theatre", name
        )

    # 📉 Step 8: Analyze least similar theatres (for contrast and verification)
    bottom_matches = similarities[-3:]

    print("\n🔻 Bottom 3 least similar theatres (lowest cosine similarity):")
    for i, (name, score) in enumerate(bottom_matches, 1):
        print(f"{i}. {name} — similarity: {score:.4f}")

    # 🖼️ Visualize least similar matches
    print("\n🧊 Visualizing least similar theatre matches:")
    for i, (name, score) in enumerate(bottom_matches, 1):
        matched_path = os.path.join(theatre_dir, name)
        matched_img = load_image(matched_path, show=False)
        matched_img = preprocess_image_for_matching(matched_img)

        combined = np.hstack((img_base, matched_img))
        combined_rgb = cv2.cvtColor(combined, cv2.COLOR_BGR2RGB)

        plt.figure(figsize=(12, 4))
        plt.imshow(combined_rgb)
        plt.title(f"Least Similar #{i}: {name} — similarity: {score:.4f}")
        plt.axis("off")
        plt.tight_layout()
        plt.show()

'''
✅ Results Analysis & Model Verification
This project demonstrates a deep learning-based pipeline for comparing the facades of Ukrainian theaters with global 
ones using visual similarity. The following steps summarize the system workflow and key insights:

🔄 Step-by-Step Pipeline
1) Image Collection & Resolution Check
The system scans folders containing Ukrainian and global theater images.
It generates a report of image sizes to ensure all inputs are valid and comparable.

2) Preprocessing Pipeline
Each image is resized to a uniform 600x400 resolution.
CLAHE is applied to improve contrast.
Gaussian blur reduces noise for cleaner feature extraction.

3) Descriptor Extraction via ResNet50
A pretrained ResNet50 (with removed classification head) extracts 2048-dimensional global feature vectors.
These embeddings encode semantic and structural characteristics of the facade.

4) Cosine Similarity Matching
Each global theater is compared with a selected Ukrainian reference (Kyiv, Odesa, or Lviv) using cosine similarity.
The top-5 most similar and bottom-3 least similar results are identified.

5) Grad-CAM Visualization
For explainability, Grad-CAM is applied to show which regions the CNN found most influential in its feature computation.
Heatmaps are generated for both similar and dissimilar pairs for interpretability.

📘 Interpretation of Grad-CAM Heatmaps
1) Grad-CAM is Local, Not Absolute
Heatmaps highlight regions that contributed most to the model’s activation — red/yellow for high importance, 
blue for low.
The visualization is model-specific and image-specific, not cross-comparable pixel by pixel.

2) No Direct Pixel Comparison Between Images
A blue region in one image and a red in another doesn’t imply direct dissimilarity or contradiction.
Grad-CAM only shows attention, not actual matched features.

3) Cosine Similarity Operates Globally
It compares overall embeddings — including parts not highlighted by Grad-CAM.
Symmetry, structure, and overall style affect similarity even if not visibly dominant in the heatmap.

4) What is a ‘Similar Region’?
If the same architectural area is highlighted in both images, it may reflect a shared design element.
If heatmaps highlight different zones, the similarity might still be valid — models can prioritize different cues.

5) Practical Recommendation
Use Grad-CAM to interpret what each image contributes to the model’s decision.
It is not a proof of similarity, but a tool for transparency and explainability.

📈 Future Improvements (Roadmap Ideas)
CLIP Model Integration
Introduce CLIP ViT-based descriptors as a more semantically aware alternative. Already discussed, not implemented.

Automatic Export of Results
Save top_matches and bottom_matches to a .csv file for further analysis or reporting.

Style-based Filtering
Add filtering by region, time period (e.g., Baroque, Neoclassicism), or country to support thematic comparisons.

Heatmap Difference Analysis
Develop quantitative tools to compute overlap or difference between Grad-CAM maps:
difference = abs(cam1 - cam2) for more rigorous visual comparison.

'''