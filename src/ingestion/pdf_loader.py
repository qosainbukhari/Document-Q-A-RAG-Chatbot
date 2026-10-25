from pathlib import Path

from langchain_core.documents import Document
from pypdf import PdfReader

from src.utils.logger import logger


class PDFLoader:

    def __init__(self, file_path: str):

        self.file_path = file_path

    def load_pdf(self):

        try:
            path = Path(self.file_path)

            if not path.exists():
                raise FileNotFoundError(
                    f"{self.file_path} not found"
                )

            if path.suffix.lower() != ".pdf":
                raise ValueError(
                    "Only PDF files are supported"
                )

            reader = PdfReader(self.file_path)

            documents = []

            for page_number, page in enumerate(reader.pages):

                text = page.extract_text()

                if text and text.strip():

                    documents.append(
                        Document(
                            page_content=text,
                            metadata={
                                "source": path.name,
                                "page": page_number + 1
                            }
                        )
                    )

            if not documents:
                raise ValueError(
                    "No extractable text found in PDF"
                )

            logger.info(
                f"Loaded {len(documents)} pages from PDF"
            )

            return documents

        except Exception as e:
            logger.error(f"PDF loading failed: {e}")
            raise