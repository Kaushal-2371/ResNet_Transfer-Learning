<div align="center">
  
# ResNet50 Transfer Learning — Flower Classification

[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![ResNet50](https://img.shields.io/badge/Backbone-ResNet50-blueviolet)](https://arxiv.org/abs/1512.03385)
[![ImageNet](https://img.shields.io/badge/Weights-ImageNet-orange)](https://image-net.org)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

*Teaching a network that's seen 1.2M images to recognise your flowers — in 10 epochs.*

</div>

---

Freeze ResNet50's ImageNet backbone (edges → textures → shapes, all learned for free), train only a lightweight head on 5 flower species. ~1M trainable params instead of 25M. Converges in minutes.

## Tech Stack
- Python
- TensorFlow / Keras 
- NumPy
- Matplotlib

## Dataset & Setup

TF's [Flower Photos](https://storage.googleapis.com/download.tensorflow.org/example_images/flower_photos.tgz) — 3,670 images across 5 classes, downloaded automatically. Split: 80/20, 180×180 input.

```bash
git clone https://github.com/Kaushal-2371/resnet-transfer-learning.git
pip install tensorflow numpy matplotlib Pillow opencv-python
jupyter notebook resnet-transfer-learnign.ipynb
```

## Predict

```python
image = cv2.resize(cv2.imread("flower.jpg"), (180, 180))
pred  = resnet_model.predict(np.expand_dims(image, axis=0))
print(class_names[np.argmax(pred)])   # e.g. "roses"
```

## Adapt to your own dataset

1. Point `data_dir` at your image folder
2. Change `Dense(5, ...)` → `Dense(N, ...)` for N classes
3. Optional — unfreeze top layers for fine-tuning:

```python
for layer in pretrained_model.layers[-20:]:
    layer.trainable = True
resnet_model.compile(optimizer=Adam(1e-5), ...)
```

## What's next

- [ ] Data augmentation (flips, zoom, rotation)
- [ ] Grad-CAM — visualise what the model is looking at
- [ ] Swap backbone to EfficientNetV2 for better accuracy
- [ ] Export to TFLite for mobile deployment

---

<div align="center">

*He et al. [Deep Residual Learning](https://arxiv.org/abs/1512.03385) · TF [Transfer Learning Guide](https://www.tensorflow.org/tutorials/images/transfer_learning)*

**⭐ Star if this helped you understand transfer learning!**

</div>
