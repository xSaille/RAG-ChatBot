from abc import ABC, abstractmethod
from pypdf import PdfReader
from typing import List, Dict
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering

class FileProcessor(ABC):
    @abstractmethod
    def extract_text(self, file_path: str) -> str:
        pass

class PDFFileProcessor(FileProcessor):
    def __init__(self, model_name: str = 'BAAI/bge-m3'):
        print("Using the CPU for embedding.")
        self.model = SentenceTransformer(model_name, device="cpu")
        print("Embedding model loaded.")

    def extract_text(self, file_path: str) -> str:
        print(f"Extracting text from: {file_path}")
        try:
            reader = PdfReader(file_path)
            text = ''
            for i, page in enumerate(reader.pages):
                page_text = page.extract_text()
                if page_text:
                    text += page_text + '\n'
            print(f"Extracted {len(text)} characters.")
            return text
        except Exception as e:
            print(f"Error reading PDF: {e}")
            return ""

    def chunk(self, text: str, distance_threshold: float = 0.6) -> List[Dict]:
        print("Starting text chunking...")
        paragraphs = [p.strip() for p in text.split('\n\n') if len(p.strip()) > 50]
        if not paragraphs:
            print("No suitable paragraphs found. Splitting by lines.")
            paragraphs = [p.strip() for p in text.split('\n') if len(p.strip()) > 50]
        if not paragraphs:
             print("Warning: Could not effectively split text. Treating as one chunk.")
             return [{'content': text, 'metadata': {'chunk_id': 0, 'length': len(text), 'cluster_id': 0}}]

        if len(paragraphs) == 1:
            print("Document contains only one paragraph. Skipping clustering.")
            return [{'content': paragraphs[0], 'metadata': {'chunk_id': 0, 'length': len(paragraphs[0]), 'cluster_id': 0}}]

        print(f"Obtained {len(paragraphs)} initial blocks.")
        print("Encoding paragraphs for clustering...")
        try:
             embeddings = self.model.encode(paragraphs, show_progress_bar=True, normalize_embeddings=True)
        except Exception as e:
             print(f"Error during paragraph encoding: {e}. Returning unclustered paragraphs.")
             return [{'content': p, 'metadata': {'chunk_id': i, 'length': len(p), 'cluster_id': -1}} 
                     for i, p in enumerate(paragraphs)]

        print("Clustering paragraphs...")
        cluster_model = AgglomerativeClustering(
            n_clusters=None, distance_threshold=distance_threshold,
            metric='cosine', linkage='average'
        )
        clusters = cluster_model.fit_predict(embeddings)
        num_clusters = len(set(clusters))
        print(f"Clustered into {num_clusters} semantic groups.")

        grouped_chunks_by_cluster: Dict[int, List[Dict]] = {}
        for i, (paragraph, cluster_id) in enumerate(zip(paragraphs, clusters)):
            cluster_id_int = int(cluster_id)
            if cluster_id_int not in grouped_chunks_by_cluster:
                grouped_chunks_by_cluster[cluster_id_int] = []
            chunk_data = {
                'content': paragraph,
                'metadata': {'original_index': i, 'length': len(paragraph), 'cluster_id': cluster_id_int}
            }
            grouped_chunks_by_cluster[cluster_id_int].append(chunk_data)

        final_chunks = []
        global_chunk_id = 0
        for cluster_id, chunks_in_cluster in grouped_chunks_by_cluster.items():
            chunks_in_cluster.sort(key=lambda x: x['metadata']['original_index'])
            combined_content = "\n\n".join([chunk['content'] for chunk in chunks_in_cluster])
            final_chunks.append({
                'content': combined_content,
                'metadata': {
                    'chunk_id': global_chunk_id, 'length': len(combined_content),
                    'cluster_id': cluster_id,
                    'original_indices': [c['metadata']['original_index'] for c in chunks_in_cluster]
                }
            })
            global_chunk_id += 1
            
        print(f"Created {len(final_chunks)} final semantic chunks.")
        return final_chunks

    def embed_chunks(self, chunks: List[Dict]) -> List[Dict]:
        print(f"Embedding {len(chunks)} chunks...")
        texts_to_embed = [chunk['content'] for chunk in chunks]
        instruction = "Represent this document for retrieval: "
        texts_with_instruction = [instruction + text for text in texts_to_embed]
        try:
            embeddings = self.model.encode(
                texts_with_instruction, normalize_embeddings=True, show_progress_bar=True
            )
            print(f"Generated {embeddings.shape[0]} embeddings.")
            for i, chunk in enumerate(chunks):
                if i < embeddings.shape[0]:
                    chunk['embedding'] = embeddings[i].tolist()
                else:
                    print(f"Warning: Mismatch embedding chunk {i}.")
                    chunk['embedding'] = None
        except Exception as e:
            print(f"Error during chunk embedding: {e}")
            for chunk in chunks:
                chunk['embedding'] = None

        return [chunk for chunk in chunks if chunk.get('embedding') is not None]


    def process_document(self, file_path: str) -> List[Dict]:
        text = self.extract_text(file_path)
        if not text: return []
        chunks = self.chunk(text)
        if not chunks: return []
        chunks_with_embeddings = self.embed_chunks(chunks)
        print(f"Successfully processed document into {len(chunks_with_embeddings)} chunks with embeddings.")
        return chunks_with_embeddings