# Age-and-Gender-Detection-system

> 🎯 A real-time computer vision application that predicts **age group** and **gender**
> using **Deep Learning + OpenCV** from a live webcam feed.

---

## 🚀 Overview

This project implements an **Age and Gender Detection System** using
**pre-trained CNN models (Caffe)** and **OpenCV’s DNN module**.

It captures live video from a webcam and predicts:
- 👨‍🦱 **Gender** → Male / Female  
- 🎂 **Age Group** → Range-based prediction (not exact age)

---

## ✨ Features

✔ Real-time webcam detection  
✔ Gender classification  
✔ Age group estimation  
✔ Pre-trained deep learning models  
✔ Lightweight & fast inference  
✔ Easy to run  

---

## 🛠️ Tech Stack

```text
Language     : Python
Library      : OpenCV, NumPy
Deep Learning: CNN (Caffe Models)
Input        : Live Webcam
Output       : Age Group + Gender

📥 Required Model Files

⚠️ IMPORTANT:
Place the following files inside the models/ folder.

🎂 Age Model
deploy_age.prototxt
age_net.caffemodel

👤 Gender Model
deploy_gender.prototxt
gender_net.caffemodel
