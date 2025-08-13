# 🩺 Glucose Monitoring Device Safety Analysis with RAG-Enhanced FOI Search  
**Post-Market Surveillance Using FDA MAUDE + FOI Narratives (2023–2024)**  

This project investigates the safety and reporting patterns of four widely used continuous glucose monitoring and insulin delivery devices — **Dexcom G6**, **Dexcom G7**, **Libre 2**, and **Omnipod 5** — using adverse event data from the **United States Food and Drug Administration’s (FDA) Manufacturer and User Facility Device Experience (MAUDE)** database.  

It combines **structured metadata analysis**, **natural language processing (NLP) of narrative text from Freedom of Information (FOI) requests**, and a **Retrieval-Augmented Generation (RAG)** chatbot to deliver interpretable, evidence-backed safety insights.  

---

## 🎯 Objectives  

- Filter MAUDE metadata for target devices (2023–2024) and remove low-signal columns.  
- Analyse structured metadata using:  
  - **Latent Dirichlet Allocation (LDA)** for topic modelling of report categories.  
  - **Isolation Forest** for anomaly detection on device/lot-level patterns.  
  - **Random Forest** and **Logistic Regression** for supervised prediction of adverse events.  
- Clean and tokenise FOI narrative text, extract high-frequency issue terms, and map them to standardised categories (e.g., Malfunction, Complaint, Medical Risk).  
- Build an integrated **RAG pipeline**:  
  - Generate sentence embeddings using **all-MiniLM-L6-v2**.  
  - Store in **Qdrant** vector database with device/year filters.  
  - Query via a **locally hosted LLaMA-based model** for context-grounded, concise answers.  
- Evaluate chatbot answer quality using **BLEU**, **ROUGE-L**, and **BERTScore**.  

---

## ⚙️ Methods  

- **Python** (Pandas, Scikit-learn, Matplotlib, Seaborn, NLTK, SentenceTransformers)  
- **Topic Modelling:** LDA to identify recurring themes in structured fields.  
- **Anomaly Detection:** Isolation Forest to detect outlier device batches.  
- **Supervised Learning:** Random Forest & Logistic Regression for adverse event classification.  
- **NLP Preprocessing:** Stopword removal, tokenisation, frequency counts, and issue-category mapping.  
- **Semantic Search:** Sentence embeddings stored in Qdrant with metadata-based filtering.  
- **RAG Chatbot:** LLaMA-based local inference integrated into a Streamlit interface for domain-specific Q&A.  

---

## 📊 Key Results  

- **Metadata Models:** Both Random Forest and Logistic Regression achieved high recall (0.99) and precision (>0.97) for minority class detection.  
- **Anomaly Detection:** Omnipod 5 flagged as an outlier in multiple time periods.  
- **FOI Narrative Analysis:** Identified dominant issue categories across devices; added them as searchable metadata for retrieval.  
- **RAG Chatbot:** Returned concise, evidence-backed answers with high semantic similarity (BERTScore 0.876) to gold-standard responses.  

---

## 🚧 Challenges  

- **Data Scale:** Initial dataset exceeded 6 million rows and 119 columns; filtering and column pruning reduced it to ~972K rows and 68 features.  
- **Processing Limits:** Year-wise file splitting and chunked FOI processing were required due to memory constraints.  
- **Evaluation Scope:** Automatic evaluation was based on a small held-out query set; larger-scale testing remains for future work.  

---

## 🔮 Future Work  

- Expand to additional device categories beyond glucose monitoring.  
- Fine-tune embeddings and language models for the medical device domain to improve retrieval precision.  
- Extend RAG chatbot with multi-modal inputs (e.g., PDF manuals, device images).  
- Integrate in-chat analytics and visualisations for regulatory and clinical decision support.  

---

## 📜 License  

This project is for academic purposes and is not intended for clinical decision-making.  
