# Data_Sci_Research
Data Science Research Project

---

# 🩺 Glucose Monitoring Device Safety Analysis (FDA MAUDE Dataset)

This project explores the safety and reporting patterns of popular glucose monitoring and insulin delivery devices using data from the FDA's MAUDE (Manufacturer and User Facility Device Experience) database. 

It applies a combination of topic modelling, anomaly detection, and supervised learning to identify trends and potential safety concerns across devices such as Dexcom G6, Dexcom G7, Libre 2, and Omnipod 5.

---

## 🧠 Objectives

- Extract and filter relevant device reports from MAUDE MDR and device datasets (2023–2024)
- Apply **LDA** to discover common themes in report categories
- Use **Isolation Forest** to detect anomalous devices or lot IDs
- Train **Random Forest** and **Logistic Regression** models to predict adverse events
- Compare and visualise model performance
- Provide insights for future safety surveillance or interface development

---

## ⚙️ Methods Used

- **Python** (Pandas, Scikit-learn, Seaborn, Matplotlib)
- **NLP** with LDA topic modelling
- **Unsupervised learning** with Isolation Forest
- **Supervised learning** with Random Forest and Logistic Regression
- Data processed in **Google Colab** due to size constraints

---

## 📊 Key Results

- High recall (0.99) and precision (0.98) for predicting adverse events
- Omnipod 5 detected as an outlier in anomaly detection
- Topic modelling revealed clear device groupings

---

## 🚧 Challenges

- Dataset size (~2.5M rows) caused repeated crashes on local machine and Colab
- Only data for 2023 and 2024 was usable due to availability in MDR reports
- Required splitting data into chunks and year-wise files

---

## 📎 Future Work

- Incorporate `foitext.txt` narrative data for full-text topic modelling
- Develop a user-friendly dashboard or chatbot interface (Part B)
- Expand to more device categories or years

---
