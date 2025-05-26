
from sentence_transformers import SentenceTransformer, util
from huggingface_hub import snapshot_download
import torch
import os
import io
import numpy as np
from typing import List, Tuple, Dict
import logging
import json
from pathlib import Path
import pickle

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class CPU_Unpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module == 'torch.storage' and name == '_load_from_bytes':
            return lambda b: torch.load(io.BytesIO(b), map_location='cpu')
        else:
            return super().find_class(module, name)

class SemanticQueryMatcher:
    """
    A class that determines if a new query is semantically similar to known supported queries.
    Uses sentence transformers for semantic similarity matching.
    """
    def __init__(self, threshold: float = 0.55, model_name: str = 'sentence-transformers/all-MiniLM-L6-v2'):
        self.threshold = threshold
        self.model_name = model_name
        self.model = None
        self.supported_queries = []
        self.supported_embeddings = None

    def _initialize_model(self):
        """Initialize the model, handling the download if needed"""
        if self.model is None:
            try:
                # First try direct initialization
                self.model = SentenceTransformer(self.model_name)
            except OSError:
                # If that fails, explicitly download the model first
                print(f"Downloading model {self.model_name}...")
                model_path = snapshot_download(repo_id=self.model_name)
                self.model = SentenceTransformer(model_path)

    def add_supported_queries(self, queries: List[str]):
        """Add supported queries and compute their embeddings."""
        self._initialize_model()  # Initialize model when needed
        self.supported_queries = queries
        self.supported_embeddings = self.model.encode(
            queries,
            convert_to_tensor=True,
            show_progress_bar=True
        )
        logger.info(f"Added {len(queries)} supported queries")

    def find_best_match(self, query: str) -> Tuple[bool, float, str]:
        """
        Find if the query matches any supported query and return the best match.

        Args:
            query: The query string to check

        Returns:
            Tuple of (is_supported, similarity_score, best_matching_query)
        """
        # Encode the query
        query_embedding = self.model.encode(query, convert_to_tensor=True)

        # Calculate cosine similarities with all supported queries
        similarities = util.cos_sim(query_embedding, self.supported_embeddings)[0]

        # Get best match
        best_match_idx = torch.argmax(similarities)
        best_match_score = similarities[best_match_idx].item()

        is_supported = best_match_score >= self.threshold
        best_matching_query = self.supported_queries[best_match_idx]

        return is_supported, best_match_score, best_matching_query

    def evaluate_queries(self, test_queries: List[str]) -> List[Dict]:
        """
        Evaluate a list of test queries and return detailed results.

        Args:
            test_queries: List of query strings to evaluate

        Returns:
            List of dictionaries containing evaluation results for each query
        """
        results = []
        for query in test_queries:
            is_supported, score, best_match = self.find_best_match(query)
            results.append({
                'query': query,
                'is_supported': is_supported,
                'similarity_score': score,
                'best_matching_query': best_match
            })
        return results

    def save(self, path: str):
        """
        Save the matcher state to disk.
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        
        # Save the model directly to the path
        self.model.save(str(path))
        
        # Save the supported queries and threshold
        state = {
            'supported_queries': self.supported_queries,
            'threshold': self.threshold
        }
        
        with open(path / 'state.json', 'w') as f:
            json.dump(state, f)
        
        # Save the supported embeddings
        with open(path / 'embeddings.pkl', 'wb') as f:
            pickle.dump(self.supported_embeddings, f)
            
        logger.info(f"Matcher saved to {path}")

    @classmethod
    def load(cls, path: str) -> 'SemanticQueryMatcher':
        """Load a matcher from disk."""
        path = Path(path)
        
        # Load the state
        with open(path / 'state.json', 'r') as f:
            state = json.load(f)
        
        # Create new instance without initializing the model
        matcher = cls(threshold=state['threshold'], model_name=str(path))
        
        # Load the model
        matcher.model = SentenceTransformer(str(path))
        
        # Load the supported queries
        matcher.supported_queries = state['supported_queries']
        
        # Load the supported embeddings
        with open(path / 'embeddings.pkl', 'rb') as f:
            
            matcher.supported_embeddings = CPU_Unpickler(f).load()
        
        logger.info(f"Matcher loaded from {path}")
        return matcher


if __name__ == "__main__":

    """Example usage of the SemanticQueryMatcher"""

    promptsFilePath = "./combined_prompts.txt"
    with open(promptsFilePath, 'r') as file:
        prompts = file.readlines()
        # Remove newline characters
        prompts = [prompt.strip() for prompt in prompts]

    # # Example supported queries
    # supported_queries = [
    #     "Why did the model predict this instance as class A?",
    #     "What are the important features for this prediction?",
    #     "Show me the feature importance for this instance",
    #     "Explain why this prediction was made",
    #     "Which features contributed most to this prediction?",
    #     "What factors influenced this classification?"
    # ]

    # Initialize matcher
    matcher = SemanticQueryMatcher()
    matcher.add_supported_queries(prompts)

    # Example test queries
    test_queries = [
        "What caused the model to make this prediction?",  # Should match
        "Why was this instance classified this way?",      # Should match
        "What's the weather like today?",                  # Shouldn't match
        "Tell me about the feature impacts",              # Should match
        "How do I make pizza?",                           # Shouldn't match
        "why skin thickness is important?"
    ]

    # Evaluate test queries
    results = matcher.evaluate_queries(test_queries)

    # Print results
    print("\nEvaluation Results:")
    print("-" * 80)
    for result in results:
        print(f"\nQuery: {result['query']}")
        print(f"Is Supported: {result['is_supported']}")
        print(f"Similarity Score: {result['similarity_score']:.4f}")
        print(f"Best Matching Query: {result['best_matching_query']}")

    matcher.save("semantic_matcher")
