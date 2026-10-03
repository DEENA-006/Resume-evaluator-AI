import base64
import os
import streamlit as st
from dotenv import load_dotenv
from parser import load_and_parse_pdf
from vector_store import chunk_documents, build_faiss_vectorstore, retrieve_relevant_chunks
from chains import run_full_evaluation_pipeline

load_dotenv()

st.set_page_config(
    page_title="AI Resume Evaluator & ATS Matcher",
    page_icon="📄",
    layout="wide" # Crucial for the side-by-side layout
)

# Resolve Gemini API Key from environment or Streamlit Secrets
api_key = os.getenv("GOOGLE_API_KEY")
if not api_key:
    try:
        if hasattr(st, "secrets") and "GOOGLE_API_KEY" in st.secrets:
            api_key = st.secrets["GOOGLE_API_KEY"]
            os.environ["GOOGLE_API_KEY"] = api_key
    except Exception:
        pass

# ==========================================
# NEW: Custom CSS Injection for Modern UI
# ==========================================
# ==========================================
# UPDATED: Custom CSS Injection for Modern UI
# ==========================================
def set_custom_ui():
    st.markdown("""
    <style>
        /* Hide top header, GitHub link/badge, Streamlit toolbar, and footer */
        header[data-testid="stHeader"] {
            display: none !important;
        }
        header {
            visibility: hidden !important;
            height: 0px !important;
        }
        [data-testid="stToolbar"] {
            display: none !important;
        }
        .stDeployButton {
            display: none !important;
        }
        #MainMenu {
            visibility: hidden !important;
            display: none !important;
        }
        footer {
            visibility: hidden !important;
            display: none !important;
        }
        div[class*="viewerBadge"] {
            display: none !important;
        }
        a[href*="github.com"] {
            display: none !important;
        }
        /* Ensure the sidebar toggle icon stays accessible */
        [data-testid="collapsedControl"] {
            visibility: visible !important;
            display: block !important;
            z-index: 999999 !important;
            top: 15px !important;
            left: 15px !important;
        }

        /* App Background */
        .stApp {
            background-color: #0F111A;
            color: #E2E8F0;
        }

        /* Main Title Gradient */
        h1 {
            background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-size: 3rem !important;
            font-weight: 800 !important;
            padding-bottom: 10px;
        }

        /* Glassmorphism for Metric Cards */
        div[data-testid="metric-container"] {
            background: rgba(255, 255, 255, 0.03);
            border: 1px solid rgba(255, 255, 255, 0.1);
            border-radius: 12px;
            padding: 20px;
            box-shadow: 0 4px 30px rgba(0, 0, 0, 0.1);
            backdrop-filter: blur(5px);
            -webkit-backdrop-filter: blur(5px);
            transition: transform 0.3s ease;
        }
        
        /* Hover effect on cards */
        div[data-testid="metric-container"]:hover {
            transform: translateY(-5px);
            border: 1px solid rgba(146, 254, 157, 0.5);
        }

        /* Primary Button Styling */
        div.stButton > button:first-child {
            background: linear-gradient(90deg, #00C9FF 0%, #92FE9D 100%);
            color: #0F111A;
            border: none;
            border-radius: 8px;
            font-weight: 700;
            padding: 0.6rem 2rem;
            transition: all 0.3s ease;
        }
        
        div.stButton > button:first-child:hover {
            box-shadow: 0 0 15px rgba(146, 254, 157, 0.6);
            transform: scale(1.02);
        }

        /* Tab Styling */
        .stTabs [data-baseweb="tab-list"] {
            gap: 24px;
        }
        .stTabs [data-baseweb="tab"] {
            height: 50px;
            white-space: pre-wrap;
            background-color: transparent;
            border-radius: 4px 4px 0px 0px;
            gap: 1px;
            padding-top: 10px;
            padding-bottom: 10px;
        }
    </style>
    """, unsafe_allow_html=True)
set_custom_ui()


# Helper function to display the PDF natively in Streamlit using pypdfium2 image rendering
def display_pdf(uploaded_file):
    try:
        import pypdfium2 as pdfium
        bytes_data = uploaded_file.getvalue()
        doc = pdfium.PdfDocument(bytes_data)
        num_pages = len(doc)
        
        st.caption(f"📄 Document contains {num_pages} page{'s' if num_pages > 1 else ''}")
        
        # Render pages as crisp images
        max_preview_pages = min(num_pages, 5)
        for page_idx in range(max_preview_pages):
            if num_pages > 1:
                st.markdown(f"**Page {page_idx + 1} of {num_pages}**")
            image = doc[page_idx].render(scale=2.0).to_pil()
            st.image(image, use_container_width=True)
            
        if num_pages > 5:
            st.info(f"Showing first 5 pages of {num_pages}. Use download button to view entire document.")
            
        st.download_button(
            label="⬇️ Download Original PDF",
            data=bytes_data,
            file_name=uploaded_file.name,
            mime="application/pdf",
            use_container_width=True
        )
    except Exception as e:
        st.warning("Visual preview could not be generated.")
        st.download_button(
            label="⬇️ Download PDF to View",
            data=uploaded_file.getvalue(),
            file_name=uploaded_file.name,
            mime="application/pdf",
            use_container_width=True
        )

st.title("📄 AI Resume Evaluator & ATS Matcher")
st.caption("Powered by Gemini 3.5 Flash, FAISS Vector Search & LangChain")

# Sidebar: Inputs
with st.sidebar:
    st.header("1. Input Data")
    
    # Allow manual API key input if not found in environment/secrets
    if not api_key:
        user_key = st.text_input(
            "🔑 Google Gemini API Key",
            type="password",
            placeholder="AIzaSy...",
            help="Enter your Google Gemini API key. It will only be used for this session."
        )
        if user_key:
            api_key = user_key
            os.environ["GOOGLE_API_KEY"] = user_key
        else:
            st.warning("⚠️ No `GOOGLE_API_KEY` detected. Please enter it above or configure it in your deployment Secrets.")

    uploaded_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"])
    job_description = st.text_area(
        "Target Job Description (JD)",
        height=260,
        placeholder="Paste role requirements, technical skills, and responsibilities..."
    )
    evaluate_btn = st.button("Evaluate Match", type="primary", use_container_width=True)

# Main Execution Flow
if evaluate_btn:
    if not api_key:
        st.error("🔑 Google Gemini API Key missing! Please enter your API key in the sidebar or configure GOOGLE_API_KEY in your deployment environment/secrets.")
    elif not uploaded_file:
        st.error("Please upload a resume PDF first.")
    elif not job_description.strip():
        st.error("Please provide a target Job Description.")
    else:
        with st.status("Analyzing Resume with Gemini...", expanded=True) as status:
            st.write("📖 Extracting and cleaning PDF text...")
            pages = load_and_parse_pdf(uploaded_file, original_filename=uploaded_file.name)
            full_text = "\n\n".join([doc.page_content for doc in pages])
            
            if not full_text.strip():
                st.error("⚠️ Could not extract text from the PDF. Please ensure the PDF has selectable text and is not a scanned image.")
                st.stop()

            st.write("✂️ Generating local embeddings and building FAISS index...")
            try:
                chunks = chunk_documents(pages, chunk_size=500, chunk_overlap=100)
                vectorstore = build_faiss_vectorstore(chunks)
            except Exception as e:
                st.warning(f"Note: Vector indexing step encountered an issue ({e}), continuing with LLM analysis...")

            st.write("🧠 Executing 4-step evaluation pipeline (Extract → Score → Coach → Cover Letter)...")
            results = run_full_evaluation_pipeline(full_text, job_description, api_key=api_key)

            status.update(label="Evaluation Finished Successfully!", state="complete", expanded=False)

        # ==========================================
        # NEW: Side-by-Side UI Layout
        # ==========================================
        col_pdf, col_results = st.columns([1, 1.2], gap="large")

        with col_pdf:
            st.subheader("Uploaded Document")
            display_pdf(uploaded_file)

        with col_results:
            st.subheader("Evaluation Results")
            profile = results["profile"]
            scores = results["scores"]
            coaching = results["coaching"]
            cover_letter = results["cover_letter"]

            # --- Metric KPI Cards ---
            match_val = int(scores.get("overall_match_score", 0))
            kpi1, kpi2, kpi3 = st.columns(3)
            kpi1.metric("Overall ATS Match", f"{match_val}%")
            kpi2.metric("Technical Fit", f"{scores.get('technical_skills_score', 0)} / 10")
            kpi3.metric("Experience Fit", f"{scores.get('experience_fit_score', 0)} / 10")

            st.progress(min(max(match_val / 100, 0.0), 1.0))
            st.info(f"**Hiring Manager Rationale:** {scores.get('rationale', 'N/A')}")

            # --- Detailed Tabs ---
            tab_skills, tab_cover, tab_coaching, tab_extracted = st.tabs([
                "🎯 Skills & Gaps",
                "📝 Auto Cover Letter",
                "💡 AI Coach",
                "📋 Extracted JSON"
            ])

            with tab_skills:
                col_match, col_miss = st.columns(2)
                with col_match:
                    st.success("### ✅ Matched Skills")
                    for skill in scores.get("matched_skills", []):
                        st.markdown(f"- **{skill}**")
                with col_miss:
                    st.error("### ⚠️ Missing Skills")
                    for gap in scores.get("missing_skills", []):
                        st.markdown(f"- **{gap}**")

            with tab_cover:
                st.markdown("### Tailored Cover Letter")
                st.markdown(cover_letter)
                # Export Button
                st.download_button(
                    label="⬇️ Download Cover Letter (.txt)",
                    data=cover_letter,
                    file_name=f"Cover_Letter_{profile.get('candidate_name', 'Candidate').replace(' ', '_')}.txt",
                    mime="text/plain"
                )

            with tab_coaching:
                st.markdown("### Actionable Resume Rewrites")
                st.markdown(coaching)

            with tab_extracted:
                st.json(profile)

else:
    st.info("👈 Upload your resume PDF and paste a Job Description in the sidebar to begin.")