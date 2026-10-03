import os
import re
import tempfile
from typing import List, Union
from io import BytesIO
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


def clean_pdf_text(raw_text: str) -> str:
    """Removes layout artifacts, broken line-breaks, and excessive whitespace."""
    text = raw_text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r'(\w+)-\n(\w+)', r'\1\2', text)
    text = re.sub(r'(?<!\n)\n(?!\n)', ' ', text)
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f-\xff]', '', text)
    return text.strip()


def load_and_parse_pdf(file_input: Union[str, BytesIO, bytes], original_filename: str = "resume.pdf") -> List[Document]:
    """Loads PDF from local path or Streamlit buffer and returns cleaned Document objects."""
    temp_file_path = None
    try:
        if isinstance(file_input, str):
            pdf_path = file_input
        else:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                if isinstance(file_input, bytes):
                    tmp_file.write(file_input)
                else:
                    tmp_file.write(file_input.getvalue())
                temp_file_path = tmp_file.name
            pdf_path = temp_file_path

        loader = PyPDFLoader(pdf_path)
        raw_documents = loader.load()

        cleaned_documents = []
        for doc in raw_documents:
            cleaned_text = clean_pdf_text(doc.page_content)
            if cleaned_text:
                metadata = dict(doc.metadata)
                metadata["source"] = original_filename
                cleaned_documents.append(
                    Document(page_content=cleaned_text, metadata=metadata)
                )

        return cleaned_documents

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            os.remove(temp_file_path)