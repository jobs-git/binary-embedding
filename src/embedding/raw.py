import fitz
import numpy as np
import io

class DocumentEmbedding:
    def __init__(self, vector_size = 384):
        self.vector_size = vector_size

    def get_raw_bytes(self, file_bytes):

        pdf = fitz.open(stream=file_bytes, filetype="pdf")
        
        new_pdf = fitz.open()
        new_pdf.insert_pdf(pdf, from_page=0, to_page=0)
        
        output = io.BytesIO()
        new_pdf.save(output)
        new_pdf.close()
        
        return output.getvalue()

    def single_encode(self, file_bytes: bytes) -> np.ndarray:
        bins = self.vector_size
        
        page_bytes = self.get_raw_bytes(file_bytes)

        byte_data = np.frombuffer(page_bytes, dtype=np.uint8)

        bin_edges = np.linspace(0, 256, num=bins + 1, dtype=np.int32)
        hist = np.histogram(byte_data, bins=bin_edges)[0].astype(np.float32)

        total = hist.sum()
        return hist / total if total > 0 else np.zeros(bins, dtype=np.float32)
    
    # TODO: Vectorize
    def encode(self, data_bytes: list[bytes]) -> list:
        embedding = []
        for file_bytes in data_bytes:
            embedding.append(self.single_encode(file_bytes))
        return embedding
