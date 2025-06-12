# 3D Knee CT Segmentation and Feature Analysis

A comprehensive pipeline for 3D knee CT image segmentation and deep learning-based feature similarity analysis using a modified DenseNet121 architecture.

## 🔬 Overview

This project implements an end-to-end solution for analyzing 3D knee CT scans through automated segmentation and feature extraction. The pipeline identifies and analyzes three key anatomical regions (tibia, femur, and background) using advanced image processing techniques combined with deep learning approaches.

## ✨ Key Features

- **3D Medical Image Segmentation**: Semi-automated segmentation of knee CT scans into anatomical regions
- **2D-to-3D Model Conversion**: Adaptation of pretrained DenseNet121 for 3D volumetric data
- **Multi-Scale Feature Extraction**: Feature extraction from multiple convolutional layers
- **Similarity Analysis**: Cosine similarity computation between anatomical regions
- **Robust Processing Pipeline**: Handles variable anatomical structures and artifacts

## 🏗️ Architecture

### Segmentation Pipeline
```
3D CT Input → Semi-Automated Processing → Watershed Algorithm → Batch Processing → Color-Coded Regions
```

### Feature Extraction
```
3D Regions → 3D DenseNet121 → Multi-Layer Hooks → Global Average Pooling → Feature Vectors
```

### Analysis
```
Feature Vectors → Cosine Similarity → CSV Output
```

## 📋 Requirements

- Python 3.8+
- PyTorch
- OpenCV
- NumPy
- SciPy
- scikit-image
- Pandas

## 🚀 Installation

```bash
git clone https://github.com/yourusername/3d-knee-ct-analysis.git
cd 3d-knee-ct-analysis
pip install -r requirements.txt
```

## 💻 Usage

### 1. Image Segmentation
```python
# Segment 3D CT into anatomical regions
from segmentation import segment_knee_ct

regions = segment_knee_ct(ct_volume)
# Returns: tibia (green), femur (red), background (gray)
```

### 2. Model Conversion
```python
# Convert 2D DenseNet121 to 3D
from model_converter import convert_2d_to_3d

model_3d = convert_2d_to_3d('densenet121')
```

### 3. Feature Extraction
```python
# Extract features from multiple layers
from feature_extractor import extract_features

features = extract_features(model_3d, regions)
# Returns: features from last, 3rd-last, 5th-last layers
```

### 4. Similarity Analysis
```python
# Compute cosine similarities
from similarity_analysis import compute_similarities

similarities = compute_similarities(features)
# Saves results to CSV file
```

## 📊 Output

The pipeline generates:
- **Segmented 3D volumes** with color-coded anatomical regions
- **Feature vectors** from multiple abstraction levels
- **Similarity matrix** (CSV format) containing:
  - Tibia vs. Femur similarity scores
  - Tibia vs. Background similarity scores  
  - Femur vs. Background similarity scores

## 🔧 Technical Details

### Segmentation Strategy
- **Semi-automated approach**: Combines algorithmic processing with visual inspection
- **Watershed algorithm**: Separates connected blob artifacts
- **Batch processing**: Adaptive thresholding for variable anatomical structures
- **Color coding**: Systematic region labeling (Green=Tibia, Red=Femur, Gray=Background)

### Model Architecture
- **Base model**: DenseNet121 (pretrained on ImageNet)
- **Conversion process**: 
  - 2D → 3D parameter mapping
  - Weight replication along depth dimension
  - Normalization by depth factor
  - 3D kernel/stride/padding configuration

### Feature Extraction
- **Multi-layer extraction**: Last, 3rd-last, 5th-last convolutional layers
- **Hook mechanism**: Captures intermediate representations during forward pass
- **Global average pooling**: Converts feature maps to fixed-size vectors
- **Three abstraction levels**: Low, mid, and high-level features

## 📈 Results

The pipeline provides quantitative similarity metrics across:
- **Region pairs**: All combinations of tibia, femur, and background
- **Feature levels**: Low-level textures to high-level semantic features
- **Volumetric analysis**: Full 3D structural comparison

## 🛠️ Customization

### Adjusting Segmentation Parameters
```python
# Modify watershed and thresholding parameters
config = {
    'min_size_threshold': 100,
    'watershed_markers': 'auto',
    'batch_size': 10
}
```

### Adding New Feature Layers
```python
# Extract from additional layers
layer_indices = ['last', '3rd_last', '5th_last', '7th_last']
```

## 📝 File Structure

```
3d-knee-ct-analysis/
├── src/
│   ├── segmentation.py          # CT segmentation pipeline
│   ├── model_converter.py       # 2D to 3D model conversion
│   ├── feature_extractor.py     # Multi-layer feature extraction
│   └── similarity_analysis.py   # Cosine similarity computation
├── data/
│   ├── input/                   # Raw CT scans
│   └── output/                  # Processed results
├── results/
│   └── similarity_scores.csv    # Analysis results
├── requirements.txt
└── README.md
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/improvement`)
3. Commit changes (`git commit -am 'Add new feature'`)
4. Push to branch (`git push origin feature/improvement`)
5. Create Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 📞 Contact

For questions or collaboration opportunities, please open an issue or contact [your-email@example.com].

## 🙏 Acknowledgments

- PyTorch team for the pretrained DenseNet121 model
- Medical imaging community for CT processing techniques
- Contributors to scikit-image and OpenCV libraries

---

**Note**: This pipeline is designed for research purposes. Clinical applications require additional validation and regulatory approval.