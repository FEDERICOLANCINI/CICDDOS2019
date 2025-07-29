🛡️ DDoS Attack Detection with CNN on CICDDoS2019 Dataset

📌 Project Overview
This project explores the application of Convolutional Neural Networks (CNNs) for detecting and classifying Distributed Denial of Service (DDoS) attacks using the CICDDoS2019 dataset, developed by the Canadian Institute for Cybersecurity.

While CNNs are traditionally used in computer vision, this work demonstrates how they can also be effectively leveraged in network traffic analysis, achieving competitive — and in some cases superior — performance compared to traditional machine learning models.

📦 Dataset
The CICDDoS2019 dataset contains a wide range of network traffic flows labeled as either benign or belonging to different DDoS attack types, such as:

SYN flood

UDP flood

HTTP flood

...and many others

Each traffic flow is described by a set of engineered features (e.g., packet count, flow duration, byte rate), extracted using CICFlowMeter.

📁 Due to its size, the dataset is not included in this repository, but can be downloaded from the official website:
👉 https://www.unb.ca/cic/datasets/ddos-2019.html

The Jupyter notebooks provided in this repo are designed to work directly with the original CSV files once downloaded.

🔍 Project Structure
This repository includes:

✅ Data preprocessing: cleaning, feature selection, normalization

📊 Exploratory Data Analysis (EDA): class distribution, feature correlation, imbalance

🧠 Modeling:

Baseline models (e.g., Logistic Regression, Random Forest)

A custom CNN architecture for tabular data

📈 Evaluation:

Performance metrics (Accuracy, Precision, Recall, F1-score, AUC)

Comparative analysis between standard models and CNN

🧠 Why a CNN?
Although CNNs are most commonly used for image data, their ability to capture local patterns and hierarchical structures can also be useful for structured/tabular data, especially when input is treated as a 2D feature map.
In this project, we reshape the input features to exploit this structure — with surprisingly effective results.

🛠️ Tech Stack
Python 3.x

Libraries: pandas, numpy, matplotlib, scikit-learn, tensorflow/keras

Jupyter Notebooks for reproducibility


✅ Results & Conclusions
The CNN model outperformed traditional models in several test scenarios, particularly in recall and F1-score for attack detection.

This highlights the potential of deep learning approaches even in non-image, tabular datasets when properly reshaped and engineered.

📚 Acknowledgments
Dataset by the Canadian Institute for Cybersecurity

Project developed as part of a broader research activity on AI for cybersecurity.


