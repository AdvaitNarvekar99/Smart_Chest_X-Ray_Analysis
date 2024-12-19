<h1 align="center">🩺 Chest Disease Detection using Deep Learning (R-CNN)</h1>  
<p align="center"><i>Accelerating Diagnosis with Artificial Intelligence</i></p>  

---

<h2>✨ Description</h2>  
<p align="justify">  
The <b>Chest Disease Detection using Deep Learning (R-CNN)</b> project utilizes state-of-the-art deep learning techniques to automate the detection and classification of chest diseases from X-ray images.<br>  
By leveraging the Faster R-CNN architecture and transfer learning, the system assists radiologists in diagnosing conditions like pneumonia, tuberculosis, and lung cancer with improved accuracy and efficiency.<br>  
The project employs the NIH ChestX-ray14 dataset, a large and annotated dataset, for robust model training and evaluation.  
</p>  

---

<h2>🎯 Objectives</h2>  
<p align="justify">  
- <b>Develop</b> a Faster R-CNN-based model for detecting thoracic pathologies.<br>  
- <b>Evaluate</b> model performance using metrics such as Average Precision (AP), mAP, sensitivity, and specificity.<br>  
- <b>Enhance</b> detection accuracy using data augmentation and transfer learning.<br>  
- <b>Visualize</b> model outputs for better interpretability by medical professionals.<br>  
</p>  

---

<h2>🛠️ Technologies</h2>  
<p align="justify">  
- <b>Deep Learning Frameworks:</b> PyTorch, Detectron2<br>  
- <b>Data Manipulation:</b> NumPy, Pandas<br>  
- <b>Visualization:</b> Matplotlib, Seaborn<br>  
- <b>Dataset:</b> NIH ChestX-ray14<br>  
- <b>Model Architecture:</b> Faster R-CNN with ResNet-50 backbone<br>  
</p>  

---

<h2>📂 Dataset</h2>  
<p align="justify">  
The project uses the <b>NIH ChestX-ray14</b> dataset, one of the largest publicly available repositories of chest X-ray images.<br>  

<b>Key Features:</b><br>  
- <b>Size:</b> Over 112,000 PNG images (1024 x 1024 pixels)<br>  
- <b>Annotations:</b> 14 disease classes, including Atelectasis, Cardiomegaly, and Pleural Effusion<br>  
- <b>Challenges:</b> Limited bounding box annotations and potential labeling inaccuracies<br>  
</p>  

---

<h2>⚙️ Methodology</h2>  

<h3>1️⃣ Data Preprocessing</h3>  
<p align="justify">  
- Resized images to fit model input dimensions (e.g., 224 x 224).<br>  
- Normalized pixel intensity values to stabilize training.<br>  
- Applied data augmentation techniques, such as flipping, rotation, and cropping.<br>  
</p>  

<h3>2️⃣ Model Development</h3>  
<p align="justify">  
- Leveraged pre-trained ResNet-50 for feature extraction.<br>  
- Fine-tuned Faster R-CNN for chest disease detection.<br>  
- Used transfer learning to reduce training time and improve performance.<br>  
</p>  

<h3>3️⃣ Evaluation</h3>  
<p align="justify">  
- Calculated AP, mAP, sensitivity, and specificity.<br>  
- Monitored validation loss to assess model generalization.<br>  
</p>  

---

<h2>📊 Results</h2>  
<p align="justify">  
- <b>AP Scores:</b> Achieved high precision and recall across most disease classes.<br>  
- <b>Detection Accuracy:</b> Accurately detected thoracic abnormalities, including subtle features often missed by human interpretation.<br>  
- <b>Visualization:</b> Bounding box predictions and confidence scores improved radiologist trust.<br>  
</p>  

---

<h2>⚠️ Challenges</h2>  
<p align="justify">  
- <b>Data Quality:</b> Limited bounding box annotations and potential NLP-derived label inaccuracies.<br>  
- <b>Class Imbalance:</b> Underrepresented disease classes required strategic sampling and loss weighting.<br>  
- <b>Overfitting:</b> Addressed with early stopping and careful hyperparameter tuning.<br>  
</p>  

---

<h2>🌟 Future Scope</h2>  
<p align="justify">  
- <b>Time-Series Integration:</b> Incorporate time-series data for real-time monitoring.<br>  
- <b>Segmentation Models:</b> Explore Mask R-CNN for pixel-level disease segmentation.<br>  
- <b>Interpretability:</b> Improve model explanations with heatmaps and saliency maps.<br>  
- <b>Dataset Expansion:</b> Include external datasets for rare disease cases to enhance generalizability.<br>  
</p>  

---
