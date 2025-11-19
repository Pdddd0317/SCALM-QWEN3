"""
Local Embedding Model Support
Use local models like sentence-transformers to replace OpenAI API

Supported Models:
1. all-MiniLM-L6-v2 (Lightweight, English, Recommended)
2. paraphrase-multilingual-MiniLM-L12-v2 (Multilingual)
3. all-mpnet-base-v2 (High Quality, English)
4. bge-large-en (From BAAI, Good Performance)
5. text2vec-base-chinese (Chinese)
"""

import numpy as np
from typing import List, Union
import os
import os
os.environ["HF_HOME"] = r"C:\Users\wangxiangyu\.cache\huggingface"
# 新增：彻底关闭 torch._dynamo，避免 substitute_in_graph 报错
os.environ["TORCHDYNAMO_DISABLE"] = "1"


# ============================================
# Local Embedding Model Class
# ============================================

class LocalEmbedding:
    """Local embedding model wrapper class"""
    
    def __init__(self, model_name='all-MiniLM-L6-v2', device='cuda'):
        """
        Initialize local embedding model
        
        Args:
            model_name: Model name
                - 'all-MiniLM-L6-v2': Lightweight, fast (Recommended)
                - 'all-mpnet-base-v2': High quality
                - 'paraphrase-multilingual-MiniLM-L12-v2': Multilingual
                - 'BAAI/bge-large-en': Good performance
                - 'shibing624/text2vec-base-chinese': Chinese
            device: 'cpu' or 'cuda'
        """
        self.model_name = model_name
        self.device = device
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load model"""
        try:
            from sentence_transformers import SentenceTransformer
            print(f"Loading local embedding model: {self.model_name}")
            print(f"Device: {self.device}")
            
            self.model = SentenceTransformer(self.model_name, device=self.device)
            
            print(f"✓ Model loaded successfully")
            print(f"  Model dimension: {self.model.get_sentence_embedding_dimension()}")
            
        except ImportError:
            print("❌ sentence-transformers not installed")
            print("Please run: pip install sentence-transformers")
            raise
        except Exception as e:
            print(f"❌ Model loading failed: {e}")
            raise
    
    def embed(self, texts: Union[str, List[str]], batch_size=32, show_progress=True) -> np.ndarray:
        """
        Generate text embedding vectors
        
        Args:
            texts: Single text or list of texts
            batch_size: Batch processing size
            show_progress: Whether to show progress bar
        
        Returns:
            numpy array with shape (n, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]
        
        # Preprocessing: replace newlines
        texts = [t.replace("\n", " ") for t in texts]
        
        # Generate embeddings
        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=show_progress,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity calculation
        )
        
        return embeddings
    
    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Make instance callable"""
        return self.embed(texts)


# ============================================
# Wrapper Functions Compatible with deep_lake
# ============================================

# Global model instance and thread lock
import threading
_global_model = None
_model_lock = threading.Lock()

def get_local_embedding_model(model_name='all-MiniLM-L6-v2', device='cpu'):
    """Get global model instance (singleton pattern, thread-safe)"""
    global _global_model
    
    # Double-checked locking
    if _global_model is None:
        with _model_lock:
            # Check again to prevent multiple threads from entering simultaneously
            if _global_model is None:
                print(f"🔒 [Thread-safe] Initializing embedding model: {model_name}")
                _global_model = LocalEmbedding(model_name=model_name, device=device)
                print(f"✅ [Thread-safe] Model initialization complete")
    
    return _global_model


def local_embedding_function(texts, model_name='all-MiniLM-L6-v2'):
    """
    Embedding function compatible with deep_lake
    
    Args:
        texts: Text or list of texts
        model_name: Model name
    
    Returns:
        numpy array
    """
    model = get_local_embedding_model(model_name=model_name)
    embeddings = model.embed(texts, show_progress=False)
    return embeddings


# ============================================
# Hybrid Mode: Support OpenAI API and Local Model
# ============================================

class HybridEmbedding:
    """Hybrid embedding: Prioritize local model, fallback to API on failure"""
    
    def __init__(self, 
                 local_model_name='all-MiniLM-L6-v2',
                 use_api_fallback=True,
                 device='cpu'):
        """
        Initialize hybrid embedding
        
        Args:
            local_model_name: Local model name
            use_api_fallback: Whether to use API when local model fails
            device: Device
        """
        self.use_api_fallback = use_api_fallback
        
        # Try to load local model
        try:
            self.local_model = LocalEmbedding(local_model_name, device)
            self.has_local_model = True
            print("✓ Will use local embedding model")
        except Exception as e:
            print(f"⚠️  Local model loading failed: {e}")
            self.has_local_model = False
            
            if use_api_fallback:
                print("   Will fallback to OpenAI API")
            else:
                raise
    
    def embed(self, texts: Union[str, List[str]]) -> np.ndarray:
        """Generate embedding vectors"""
        # Prioritize local model
        if self.has_local_model:
            try:
                return self.local_model.embed(texts, show_progress=False)
            except Exception as e:
                print(f"⚠️  Local model failed: {e}")
                if not self.use_api_fallback:
                    raise
        
        # Fallback to API
        if self.use_api_fallback:
            print("   Using OpenAI API...")
            from deep_lake import embedding_function as api_embedding
            return api_embedding(texts)
        else:
            raise RuntimeError("Local model failed and API fallback not enabled")
    
    def __call__(self, texts: Union[str, List[str]]) -> np.ndarray:
        return self.embed(texts)


# ============================================
# Model Recommendation and Testing
# ============================================

def recommend_model(language='en', quality='medium'):
    """
    Recommend suitable model
    
    Args:
        language: 'en', 'zh', 'multilingual'
        quality: 'fast', 'medium', 'high'
    
    Returns:
        Recommended model name
    """
    recommendations = {
        'en': {
            'fast': 'all-MiniLM-L6-v2',           # 80MB, fast
            'medium': 'all-mpnet-base-v2',        # 420MB, balanced
            'high': 'BAAI/bge-large-en'           # 1.3GB, high quality
        },
        'zh': {
            'fast': 'shibing624/text2vec-base-chinese',
            'medium': 'BAAI/bge-base-zh',
            'high': 'BAAI/bge-large-zh'
        },
        'multilingual': {
            'fast': 'paraphrase-multilingual-MiniLM-L12-v2',
            'medium': 'paraphrase-multilingual-mpnet-base-v2',
            'high': 'intfloat/multilingual-e5-large'
        }
    }
    
    return recommendations.get(language, {}).get(quality, 'all-MiniLM-L6-v2')


def test_local_embedding():
    """Test local embedding model"""
    print("=" * 80)
    print("Testing Local Embedding Model")
    print("=" * 80 + "\n")
    
    # Test texts
    test_texts = [
        "function transfer(address to, uint amount) public { balances[msg.sender] -= amount; }",
        "function approve(address spender, uint amount) public returns (bool) { return true; }",
        "function balanceOf(address account) public view returns (uint) { return balances[account]; }"
    ]
    
    # Test local model
    print("1. Testing local model (all-MiniLM-L6-v2)")
    print("-" * 80)
    
    try:
        model = LocalEmbedding('all-MiniLM-L6-v2')
        embeddings = model.embed(test_texts)
        
        print(f"✓ Successfully generated embeddings")
        print(f"  Shape: {embeddings.shape}")
        print(f"  Dimension: {embeddings.shape[1]}")
        print(f"  Sample vector first 5 values: {embeddings[0][:5]}")
        
        # Calculate similarity
        from sklearn.metrics.pairwise import cosine_similarity
        sim_matrix = cosine_similarity(embeddings)
        
        print(f"\nSimilarity matrix:")
        print(f"  transfer vs approve: {sim_matrix[0][1]:.4f}")
        print(f"  transfer vs balanceOf: {sim_matrix[0][2]:.4f}")
        print(f"  approve vs balanceOf: {sim_matrix[1][2]:.4f}")
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        return False
    
    print("\n" + "=" * 80)
    print("✅ Local embedding model test passed!")
    print("=" * 80)
    
    return True

def compare_models():
    """Compare different model performance"""
    print("\n" + "=" * 80)
    print("Comparing Different Embedding Models")
    print("=" * 80 + "\n")
    
    test_text = "function transfer(address to, uint amount) public"
    
    models_to_test = [
        'all-MiniLM-L6-v2',
        'all-mpnet-base-v2',
    ]
    
    import time
    
    for model_name in models_to_test:
        print(f"\nTesting model: {model_name}")
        print("-" * 60)
        
        try:
            # Load model
            start_time = time.time()
            model = LocalEmbedding(model_name)
            load_time = time.time() - start_time
            
            # Generate embedding
            start_time = time.time()
            embedding = model.embed(test_text)
            embed_time = time.time() - start_time
            
            print(f"✓ Model dimension: {embedding.shape[1]}")
            print(f"  Load time: {load_time:.2f}s")
            print(f"  Embedding time: {embed_time:.3f}s")
            
        except Exception as e:
            print(f"❌ Failed: {e}")


if __name__ == '__main__':
    print(__doc__)
    
    # Run tests
    print("\nStarting tests...\n")
    
    # Test basic functionality
    success = test_local_embedding()
    
    if success:
        # Compare different models
        choice = input("\nCompare different models? (y/n): ").strip().lower()
        if choice == 'y':
            compare_models()

