import unittest
from unittest.mock import Mock, patch
import numpy as np
from rag.rag_logic import RAGChatbot

class TestRAGChatbot(unittest.TestCase):
    def setUp(self):
        self.chatbot = RAGChatbot()
    
    def test_embed_query(self):
        test_query = "test query"
        mock_embedding = np.array([0.1, 0.2, 0.3])
        
        with patch.object(self.chatbot.file_processor.model, 'encode') as mock_encode:
            mock_encode.return_value = np.array([mock_embedding])
            result = self.chatbot.embed_query(test_query)
            
            self.assertIsNotNone(result)
            np.testing.assert_array_equal(result, mock_embedding)
            mock_encode.assert_called_once()
    
    def test_retrieve_relevant_chunks(self):
        query_embedding = np.array([0.1, 0.2, 0.3])
        test_chunks = [
            {'content': 'test1', 'embedding': np.array([0.1, 0.2, 0.3])},
            {'content': 'test2', 'embedding': np.array([0.4, 0.5, 0.6])}
        ]
        
        self.chatbot.processed_data = test_chunks
        result = self.chatbot.retrieve_relevant_chunks(query_embedding, top_k=1)
        
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0], 'test1')

if __name__ == '__main__':
    unittest.main()