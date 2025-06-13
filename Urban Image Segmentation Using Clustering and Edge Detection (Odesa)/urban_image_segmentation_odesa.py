# --------------------------- 🔹 Urban Image Segmentation — Odesa Project ------------------------------------
"""
Author: Ivan Tatarchuk

This script performs unsupervised image segmentation on a high-resolution urban photo of Odesa,
capturing the area around the city's Opera Theatre and Black Sea port. The primary goal is to explore
the use of computer vision and clustering techniques to extract and identify visual patterns
in a complex urban environment.

Key Objectives:
- Segment urban objects based on color clusters using the KMeans algorithm
- Improve visual clarity and contrast using image enhancement techniques (CLAHE, HSV boosts)
- Detect geometric shapes and contours using edge detection and vectorization
- Analyze how segmentation parameters (e.g., k in KMeans) impact object isolation

This project demonstrates the potential for automated recognition of architectural features,
infrastructure elements, and natural zones in aerial or street-level urban images.
Applicable for smart city analytics, remote monitoring, and visual geospatial insights.

Tested With:
- numpy==1.26.4
- pillow==11.12.1
- matplotlib==3.10.1
- scikit-learn==1.6.1
- opencv-python==4.11.0.86
"""

# --------------------------- 🔹 Imports & Dependencies ------------------------------------

from PIL import Image, ImageEnhance
import numpy as np
import matplotlib.pyplot as plt
from pylab import contour, axis, show
import cv2
from sklearn.cluster import KMeans

# Suppress specific warnings to improve readability of output
import warnings
warnings.filterwarnings("ignore", category=UserWarning)

# ------------------------ 🔹 Image Loading & Display Utilities ------------------------

def load_image(path: str, resize_to: tuple = None):
    """
    Loads an image from the specified file path and optionally resizes it.

    Args:
        path (str): Path to the image file.
        resize_to (tuple): (width, height) if resizing is needed; otherwise None.

    Returns:
        PIL.Image: RGB image object.
    """
    image = Image.open(path).convert("RGB")
    if resize_to:
        image = image.resize(resize_to)
    return image

def show_image(image, title="Image"):
    """
    Displays a PIL image using matplotlib.

    Args:
        image (PIL.Image): The image to display.
        title (str): Plot title.
    """
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(title)
    plt.axis("off")
    plt.show()

# ------------------------ 🔹 Image Segmentation using KMeans ------------------------

def segment_image_kmeans(image, n_clusters=5):
    """
    Segments an image by grouping pixels into color clusters using the KMeans algorithm.

    Args:
        image (PIL.Image): Input image in RGB format.
        n_clusters (int): Number of color clusters to segment into.

    Returns:
        PIL.Image: Segmented image with clustered colors.
    """
    img_array = np.array(image)
    w, h, d = img_array.shape
    pixels = img_array.reshape(-1, 3)

    # Perform KMeans clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = kmeans.fit_predict(pixels)
    segmented_pixels = kmeans.cluster_centers_[labels].astype("uint8")

    # Reshape clustered pixels back into image
    segmented_img = segmented_pixels.reshape(w, h, 3)
    return Image.fromarray(segmented_img)

# ------------------------ 🔹 Advanced Image Enhancement Pipeline ------------------------

def enhance_image_quality_advanced(image: np.ndarray) -> np.ndarray:
    """
    Applies advanced image enhancement techniques to improve visual clarity and separability.

    Enhancement steps:
    - CLAHE (Contrast Limited Adaptive Histogram Equalization) in LAB color space
    - Saturation and brightness boost in HSV space

    Args:
        image (np.ndarray): Input RGB image as a NumPy array.

    Returns:
        np.ndarray: Enhanced RGB image.
    """
    # Step 1: CLAHE enhancement in LAB color space
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    lab_enhanced = cv2.merge((cl, a, b))
    image_clahe = cv2.cvtColor(lab_enhanced, cv2.COLOR_LAB2RGB)

    # Step 2: Boost saturation and brightness in HSV space
    hsv = cv2.cvtColor(image_clahe, cv2.COLOR_RGB2HSV).astype("float32")
    h, s, v = cv2.split(hsv)
    s *= 1.25
    v *= 1.20
    s = np.clip(s, 0, 255)
    v = np.clip(v, 0, 255)
    hsv_enhanced = cv2.merge([h, s, v]).astype("uint8")
    final = cv2.cvtColor(hsv_enhanced, cv2.COLOR_HSV2RGB)

    return final

# ------------------------ 🔹 Geometric Segmentation via Edges & Contours ------------------------

def vector_and_canny_segmentation(image_path: str):
    """
    Performs geometric segmentation by combining:
    - Edge detection using OpenCV's Canny algorithm
    - Contour vectorization using matplotlib

    Args:
        image_path (str): Path to the image file (must be in RGB format).
    """

    # Step 1: Edge detection using Canny
    img = cv2.imread(image_path)
    canny = cv2.Canny(img, 100, 200)
    plt.figure()
    plt.imshow(canny, cmap='gray')
    plt.title("Canny Edge Detection")
    plt.axis('off')
    plt.show()

    # Step 2: Contour vectorization from grayscale version
    gray_img = np.array(Image.open(image_path).convert('L'))
    plt.figure()
    contour(gray_img, origin='image')
    plt.axis('equal')
    plt.title("Vectorized Contours (all levels)")
    show()

# -------------------------------- 🔹 MAIN PIPELINE --------------------------------

if __name__ == "__main__":
    image_path = "Odesa_image.jpg"

    # Load original image
    image = load_image(image_path, resize_to=None)
    width, height = image.size
    show_image(image, f"Original Image ({width}x{height})")

    # Step 1: KMeans segmentation on the original image
    k = 6
    segmented = segment_image_kmeans(image, n_clusters=k)
    show_image(segmented, f"KMeans Segmentation (k={k})")

    # Step 2: Enhance image quality (contrast, saturation, brightness)
    image_cv = np.array(image)  # Convert PIL → NumPy RGB
    enhanced = enhance_image_quality_advanced(image_cv)
    show_image(Image.fromarray(enhanced), f"Enhanced Image ({width}x{height})")

    # Step 3: KMeans segmentation on the enhanced image
    k = 9
    segmented_enhanced = segment_image_kmeans(Image.fromarray(enhanced), n_clusters=k)
    show_image(segmented_enhanced, f"KMeans After Enhancement (k={k})")

    # Step 4: Geometric segmentation — edges and contours
    vector_and_canny_segmentation(image_path)

"""
📊 Analysis of Results — Interpretation & Validation
-----------------------------------------------------

📌 Part 1: Color-Based Clustering on the Original Image
--------------------------------------------------------
We conducted a series of experiments using different numbers of clusters (k) in the KMeans algorithm to evaluate 
how well the model segments color regions in the image.

Key insights by k:
- k = 2: Too coarse — only splits into light vs dark zones. Some logic emerges (buildings + sea + sky vs trees + road).
- k = 3: Water and sky begin separating from buildings. Fountains and rooftops enter the "sky" cluster.
- k = 4: Trees and soil split, but autumn tones make it difficult to distinguish from buildings.
- k = 5: Sea and sky are now clearly separated; some haze in background merges with the sea.
- k = 6: Clouds and sky are distinguished clearly. Best trade-off between detail and interpretability.
- k = 7–10: No significant improvement. Over-segmentation reduces clarity.

✅ Conclusion:
k = 6 provides the best balance. Clearly separated:
→ Buildings  
→ Sky  
→ Clouds  
→ Trees, soil, and sea are moderately segmented but still visually meaningful.

🎯 Summary:
KMeans clustering effectively separates meaningful visual zones in urban imagery and serves as a foundation 
for further geometry-based detection (e.g., buildings, sea, port elements, vegetation).
--------------------------------------------------------

📌 Part 2: Enhancement for Complex Object Separation
--------------------------------------------------------
The next goal was to differentiate visually similar but semantically distinct areas:
🔸 Sea  
🔸 Distant city (covered in haze)  
🔸 Oil tankers in the port  

In the original k=6 segmentation, these objects shared color clusters due to similar tones.
To improve separation, we applied a custom `enhance_image_quality_advanced()` function with:
- CLAHE (local contrast enhancement)
- Brightness and saturation boost in HSV space

🧪 After enhancement + increasing k to 9:
✅ Partial success achieved:
• A group of tankers near the dock is clearly separated from the sea  
• An isolated large tanker was successfully grouped with other tankers  
• Some blending remains between the sea and hazy city background  

💡 Conclusion:
Even without shape-based segmentation, visual enhancement and color clustering alone help identify complex structures.
Further refinement is possible via geometric segmentation (contours, vectorization).
In this case, k = 9 yielded the most nuanced result.
--------------------------------------------------------

📌 Part 3: Geometric Object Detection (Contours + Shape Features)
--------------------------------------------------------
Finally, we applied geometric segmentation to detect visual objects based on their shape and structure.

Techniques used:
- `cv2.Canny` for edge detection  
- `matplotlib.contour` for vectorized outlines  

🔍 Visual Results:
✅ The following objects were clearly distinguished:
• Sea vs land, including detailed port infrastructure  
• Architectural details of the Odesa Opera Theatre  
• Left-side park zones and nearby cars  
• Circular fountain to the right of the theatre  

🎯 Summary:
The combination of Canny and contour vectorization reveals geometric structures that are hard to distinguish via 
color alone.
This enables further use in semantic segmentation or object detection, such as fountains, vehicles, and 
architectural forms.

Overall, this final stage significantly enhanced image clarity and completed the identification 
of the key visual elements in the chosen urban scene.
--------------------------------------------------------
"""
