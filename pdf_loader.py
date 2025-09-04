from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter

def load_and_split_pdf(file) -> list:
    """
    Load PDF and split into chunks
    """
    reader = PdfReader(file)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text

    # Optimized chunk size for speed & accuracy
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
    texts = text_splitter.split_text(text)
    return texts
