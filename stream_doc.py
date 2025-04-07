from pydantic import BaseModel
from tika import parser
import logging
from typing import List, Dict
import os

TIKA_SERVER_URL = "http://127.0.0.1:9998/tika"

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DocumentData(BaseModel):
    """
    Pydantic model for structured document data.
    """

    content: str
    metadata: Dict


class DocumentProcessor:
    """
    Processes documents by parsing them with Apache Tika.
    """

    def __init__(self, file_path: str):
        self.file_path = file_path

    def stream_document(self, file_name: str) -> Dict:
        """
        Parses the document using Apache Tika and returns a dictionary,
        containing the content and metadata.
        """
        try:
            parsed = parser.from_file(
                os.path.join(self.file_path, file_name), serverEndpoint=TIKA_SERVER_URL
            )
        except Exception as e:
            logger.error(f"Error parsing file {file_name}: {e}")
            return {}

        if "resourceName" in parsed["metadata"]:
            if isinstance(parsed["metadata"]["resourceName"], list):
                decoded_text = parsed["metadata"]["resourceName"][0].strip("b'")
            else:
                decoded_text = parsed["metadata"]["resourceName"].strip("b'")

            parsed["metadata"]["file_name"] = decoded_text
            del parsed["metadata"]["resourceName"]

        content = parsed["content"]
        metadata = parsed["metadata"]

        document_data = DocumentData(content=content, metadata=metadata)
        return document_data.model_dump()

    def stream_all_documents(self) -> List[Dict]:
        """
        Processes all documents in the directory and returns a list of dictionaries.
        """
        all_document_data: List[Dict] = []
        if os.path.isdir(self.file_path):
            for file_name in os.listdir(self.file_path):
                if os.path.isfile(os.path.join(self.file_path, file_name)):
                    document_data = self.stream_document(file_name)
                    if document_data:
                        all_document_data.append(document_data)
        else:
            logger.error(f"{self.file_path} is not a directory")
        return all_document_data


if __name__ == "__main__":
    # Beispielhafte Verwendung
    file_path = "data"  # Jetzt ein Verzeichnis
    processor = DocumentProcessor(file_path)
    result = processor.stream_all_documents()

    if result:
        print(result)
    else:
        print("No data returned.")
