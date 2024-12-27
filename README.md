# **Character Insight Extraction Using LLM (DeepStack Assignment)**

## **Demo Video**
https://github.com/user-attachments/assets/9e318c7e-04e3-4533-97b4-cee93678fd52

## **Technologies Used**
### **1. LangChain and Python**
- **LangChain**:
  - Provides a robust framework for LLM integration, chaining prompts, and managing workflows..
- **Python**:
  - Core language used for implementing the project, ensuring flexibility and integration with ML frameworks.

### **2. Large Language Models (LLMs)**
- **Model**: `mixtral-8x7b` (Inferenced using GroqAPI)
- Used for generating summaries, identifying relationships, and classifying character types.

### **3. Vector Databases**
- **ChromaDB**: Stores and retrieves text embeddings for efficient context-based search.
- **Embeddings**: Generated using HuggingFace's `all-MiniLM-L6-v2` model for high-quality semantic understanding.

### **4. Text Splitting**
- **RecursiveCharacterTextSplitter**: Splits large text into manageable chunks with overlaps to ensure continuity of context.



---
## **File Structure**
![file_structure](./output/folder_structure.PNG) 
---

## **Installation**
1. Clone the repository:
   ```bash
   git clone https://github.com/VivekShinde7/Character_Insight_Extractor_Using_LLM.git
   ```
   ```bash
   cd Audio-DeepFake-Detection-using-CNN-BiLSTM
   ```

2. Set Up a Python Environment:
   ```bash
    conda create -prefix ./env python=3.9 -y
    ```
    ```bash
    conda activate ./env
   ```
3. Install Dependencies:
   ```bash
    pip install -r requirements.txt
   ```
4. Set Up Environment Variables:
   - Create a `.env` file in the project root:
   ```bash
    HF_TOKEN = your_huggingface_token
    GROQ_API_KEY = your_groq_api_key
   ```
   - Use my API Keys if you want
   ```bash
    HF_TOKEN = "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"
    GROQ_API_KEY = "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"
   ```

## **Usage**

### 1. Preprocess the Text
- Use the `compute_embeddings.py` script to process your text and store embeddings in the vector database:
    ```bash
    python src/compute_embeddings.py data/
    ```

### 2. Analyze a Character
- Run the `get_character_info.py` script to extract character details:
    ```bash
    python src/get_character_info.py "<character_name>"
    ```
- Example 
    ```bash
    python src/get_character_info.py "Eliza"
    ```
- Output
![output](./output/output_eliza.PNG) 

## **Future Enhancements**

- **Integration with Neo4j**: Enable graph-based relationship visualization to provide a clearer and more interactive representation of character connections.
- **Fine-Tuning the LLM**: Improve role classification and relationship detection accuracy by fine-tuning the LLM on a domain-specific dataset.

