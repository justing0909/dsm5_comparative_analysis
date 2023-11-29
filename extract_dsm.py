"""
Extracts and preprocesses text from DSM
"""

# Import statements
import PyPDF2
import re
import en_core_web_sm

# Some words that do not lend itself to much analysis
# "diso" "rder" comes from page/line breaks, is not caught by parser
DSM_STOPWORDS = ["disorder", "major", "attack", "episode", "symptom", "criterion", "diso", "rder"]


def extract_text(pdf, page_range, stopwords=DSM_STOPWORDS):
    """
    Extracts a section from DSM and preprocesses the text
    :param pdf: Path to pdf file
    :param page_range: Range of pages
    :return: string of preprocessed text
    """
    # Open the PDF file and create a PDF reader object
    with open(pdf, "rb") as pdf_file:
        pdf_reader = PyPDF2.PdfFileReader(pdf_file, strict=False)

        # Extract the text from the specified pages
        text = ""

        for page_num in range(*page_range):
            page = pdf_reader.getPage(page_num)
            page_text = page.extractText()
            # Remove line breaks and page breaks
            page_text = re.sub(r"\n|\x0c", " ", page_text)
            # Remove hyphens at line breaks
            page_text = re.sub(r"(?<=\w)-\s+(?=[a-zA-Z])", "", page_text)
            text += page_text

        # Preprocess the text
        nlp = en_core_web_sm.load()
        nlp.max_length = 5000000
        doc = nlp(text)
        words = [token.lemma_ for token in doc if not token.is_stop and token.is_alpha and len(token.lemma_) > 2]
        words = [word.lower() for word in words if word.lower() not in stopwords]
        
        return " ".join(words)
