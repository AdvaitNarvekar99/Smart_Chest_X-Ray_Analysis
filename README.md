<h1 align="center">🩺 Chest Disease Detection using Deep Learning (R-CNN)</h1>  
<p align="center"><i>Accelerating Diagnosis with Artificial Intelligence</i></p>  

---

<h2>✨ Description</h2>  
<p align="justify">  
The <b>Chest Disease Detection using Deep Learning (R-CNN)</b> project utilizes advanced deep learning techniques to automate the detection and classification of chest diseases from X-ray images. By leveraging the Faster R-CNN architecture and transfer learning, the system supports radiologists in diagnosing conditions like pneumonia, tuberculosis, and lung cancer. The project is built using the NIH ChestX-ray14 dataset, which provides annotated data for robust training and evaluation.  
</p>  

---

<h2>🎯 Objectives</h2>  
<p align="justify">  
- Design a Faster R-CNN-based model for thoracic pathology detection.<br>  
- Utilize evaluation metrics such as Average Precision (AP), sensitivity, and specificity.<br>  
- Apply data augmentation and transfer learning to improve accuracy.<br>  
- Provide interpretable outputs through bounding box predictions.<br>  
</p>  

---

<h2>🛠️ Technologies Used</h2>  

<h3>💻 Frameworks and Libraries</h3>  
<p align="justify">  
- <b>PyTorch:</b> Implemented Faster R-CNN architecture with support for fine-tuning and transfer learning.<br>  
- <b>Detectron2:</b> Used for advanced object detection tasks, including bounding box predictions.<br>  
- <b>NumPy & Pandas:</b> Managed and preprocessed large-scale tabular and image datasets.<br>  
</p>  

<h3>📊 Data Visualization</h3>  
<p align="justify">  
- <b>Matplotlib:</b> Plotted results such as loss curves, AP metrics, and data distributions.<br>  
- <b>Seaborn:</b> Generated detailed bar charts and heatmaps to analyze dataset characteristics.<br>  
</p>  

<h3>📂 Dataset and Storage</h3>  
<p align="justify">  
- <b>NIH ChestX-ray14:</b> Large-scale dataset with over 112,000 labeled X-ray images.<br>  
- <b>Local and Cloud Storage:</b> Efficient storage and management of high-resolution X-ray images during training.<br>  
</p>  

<h3>⚙️ Model Development</h3>  
<p align="justify">  
- <b>ResNet-50:</b> Used as the backbone for Faster R-CNN for feature extraction.<br>  
- <b>Transfer Learning:</b> Fine-tuned pre-trained weights to adapt to chest disease detection.<br>  
</p>  

<h3>🔐 Tools and Utilities</h3>  
<p align="justify">  
- <b>Google Colab:</b> Leveraged GPU acceleration for efficient model training.<br>  
- <b>OpenCV:</b> Preprocessed X-ray images to standardize dimensions and formats.<br>  
</p>  

---

<h2>📂 Dataset</h2>  
<p align="justify">  
The <b>NIH ChestX-ray14</b> dataset provides labeled X-ray images for 14 disease classes.  

<b>Key Features:</b><br>  
- <b>Size:</b> Over 112,000 PNG images.<br>  
- <b>Annotations:</b> Diseases like Atelectasis, Cardiomegaly, and Pneumothorax.<br>  
- <b>Challenges:</b> Class imbalance and limited bounding box annotations.<br>  
</p>  

---

<h2>📈 Results</h2>  

<h3>1️⃣ Loss Curve</h3>  
<p align="center">
<img src="./LossCurve.png" alt="Loss Curve" width="700">
</p>  
<p align="justify">  
The loss curve demonstrates the model's training progress and convergence over epochs.
</p>

<h3>2️⃣ Bounding Box Predictions</h3>  
<p align="center">
<img src="./multipleIpred.png" alt="Bounding Box Predictions" width="700">
</p>  
<p align="justify">  
The Faster R-CNN model accurately localizes and classifies chest diseases, visualized using bounding box predictions.
</p>

<h3>3️⃣ Prediction Table</h3>  
<p align="center">
<img src="./ImagePredictionTable.png" alt="Prediction Table" width="700">
</p>  
<p align="justify">  
Quantitative output of the model for selected images, showcasing bounding box coordinates and confidence scores.
</p>

---

<h2>🌟 Future Scope</h2>  
<p align="justify">  
- <b>Real-Time Data:</b> Integrate APIs for real-time data streams for faster diagnostics.<br>  
- <b>Explainable AI:</b> Enhance interpretability using saliency maps and heatmaps.<br>  
- <b>Pixel-Level Segmentation:</b> Adopt Mask R-CNN to generate detailed region-based disease detections.<br>  
- <b>Generalization:</b> Expand dataset with rare diseases and global datasets.<br>  
</p>  

---
