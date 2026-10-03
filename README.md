# 📄 AI Resume Evaluator & ATS Matcher

An AI-powered ATS (Applicant Tracking System) resume analyzer, matcher, and career coach built using **Streamlit**, **Google Gemini Flash**, **FAISS Vector Search**, and **LangChain**.

---

## 🚀 Features

- **PDF Resume Extraction & Parsing**: Cleanly extracts and cleans structured text from candidate PDF resumes.
- **FAISS Vector Indexing**: Semantic chunking and vector storage using lightweight embeddings.
- **ATS Match Scoring**: Analyzes candidate qualifications against a job description to calculate:
  - Overall ATS Match Percentage (%)
  - Technical Skills Alignment Score (/10)
  - Experience Fit Score (/10)
  - Hiring Manager Rationale
- **Matched vs. Missing Skills**: Side-by-side gap analysis.
- **Actionable AI Resume Rewriting**: Rewrites bullet points using high-impact metrics and action verbs.
- **Automated Cover Letter Generation**: Generates a tailored 3–4 paragraph cover letter available for instant `.txt` export.
- **Side-by-Side Dual View**: View your uploaded resume document alongside real-time AI evaluation results.

---

## 🛠️ Tech Stack

- **Frontend**: [Streamlit](https://streamlit.io/)
- **LLM**: Google Gemini (via `langchain-google-genai` & `google-genai`)
- **Embeddings & Vector Search**: HuggingFace (`all-MiniLM-L6-v2`) / Google Embeddings & FAISS
- **PDF Loader**: `pypdf` via LangChain Community loaders

---

## ☁️ Deployment on Streamlit Community Cloud

1. Fork or push this repository to GitHub.
2. Go to [share.streamlit.io](https://share.streamlit.io/) and click **New App**.
3. Select your repository: `DEENA-006/Resume-evaluator-AI`
4. Set:
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Under **Advanced Settings** ➔ **Secrets**, add your Google Gemini API Key:
   ```toml
   GOOGLE_API_KEY = "your-gemini-api-key-here"
   ```
6. Click **Deploy**!

---

## 💻 Local Setup & Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/DEENA-006/Resume-evaluator-AI.git
   cd Resume-evaluator-AI
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv venv
   # On Windows:
   .\venv\Scripts\activate
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables:**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_gemini_api_key_here
   ```

5. **Run the Streamlit application:**
   ```bash
   streamlit run app.py
   ```
