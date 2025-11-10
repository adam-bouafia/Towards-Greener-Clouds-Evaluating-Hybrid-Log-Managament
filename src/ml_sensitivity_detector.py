"""
Lightweight ML-based Sensitivity Detection using Pre-trained Models

This module provides multiple options for using small, pre-trained models
to detect log sensitivity without training from scratch.
"""

import logging
from typing import Dict, Any, Optional
from pathlib import Path

logger = logging.getLogger(__name__)


class MLSensitivityDetector:
    """
    Machine Learning-based sensitivity detection using lightweight models.
    
    Supports multiple backends:
    1. DistilBERT (zero-shot classification) - RECOMMENDED
    2. TinyBERT (fine-tunable)
    3. sklearn LogisticRegression (classical ML)
    """
    
    def __init__(
        self,
        model_type: str = "distilbert",
        model_path: Optional[str] = None,
        confidence_threshold: float = 0.7
    ):
        """
        Initialize ML sensitivity detector.
        
        Args:
            model_type: "distilbert", "tinybert", or "sklearn"
            model_path: Path to fine-tuned model (optional)
            confidence_threshold: Minimum confidence for positive prediction
        """
        self.model_type = model_type
        self.confidence_threshold = confidence_threshold
        self.model = None
        self.tokenizer = None
        
        self._load_model(model_type, model_path)
    
    def _load_model(self, model_type: str, model_path: Optional[str]):
        """Load the specified model."""
        if model_type == "distilbert":
            self._load_distilbert(model_path)
        elif model_type == "tinybert":
            self._load_tinybert(model_path)
        elif model_type == "sklearn":
            self._load_sklearn(model_path)
        else:
            raise ValueError(f"Unknown model type: {model_type}")
    
    def _load_distilbert(self, model_path: Optional[str]):
        """
        Load DistilBERT for zero-shot classification.
        
        Model: distilbert-base-uncased-finetuned-sst-2-english
        Size: 66MB
        Speed: ~20ms per prediction
        
        No training needed - works out of the box!
        """
        try:
            from transformers import pipeline
            
            if model_path:
                # Load fine-tuned model
                self.model = pipeline(
                    "text-classification",
                    model=model_path,
                    device=-1  # CPU
                )
            else:
                # Use zero-shot with security labels
                self.model = pipeline(
                    "zero-shot-classification",
                    model="facebook/bart-large-mnli",  # 1.6GB but accurate
                    device=-1
                )
                # Alternative: Use smaller model
                # self.model = pipeline(
                #     "text-classification",
                #     model="distilbert-base-uncased-finetuned-sst-2-english",
                #     device=-1
                # )
            
            logger.info(f"Loaded DistilBERT model for sensitivity detection")
            
        except ImportError:
            logger.error("transformers not installed: pip install transformers torch")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load DistilBERT: {e}")
            self.model = None
    
    def _load_tinybert(self, model_path: Optional[str]):
        """
        Load TinyBERT (smaller, faster alternative).
        
        Model: huawei-noah/TinyBERT_General_4L_312D
        Size: 14MB
        Speed: ~5ms per prediction
        """
        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
            
            if model_path:
                model_name = model_path
            else:
                model_name = "huawei-noah/TinyBERT_General_4L_312D"
            
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModelForSequenceClassification.from_pretrained(model_name)
            self.model.eval()
            
            logger.info(f"Loaded TinyBERT model for sensitivity detection")
            
        except ImportError:
            logger.error("transformers not installed: pip install transformers torch")
            self.model = None
        except Exception as e:
            logger.error(f"Failed to load TinyBERT: {e}")
            self.model = None
    
    def _load_sklearn(self, model_path: Optional[str]):
        """
        Load sklearn model (fastest, smallest).
        
        Model: Logistic Regression + TF-IDF
        Size: <1MB
        Speed: <1ms per prediction
        
        Requires training on labeled data.
        """
        try:
            import pickle
            
            if not model_path:
                logger.warning("sklearn model requires training data")
                return
            
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            logger.info(f"Loaded sklearn model from {model_path}")
            
        except Exception as e:
            logger.error(f"Failed to load sklearn model: {e}")
            self.model = None
    
    def predict(self, log: Dict[str, Any]) -> tuple[bool, float]:
        """
        Predict if log is sensitive using ML model.
        
        Args:
            log: Log entry dict with Level, Content, Component
        
        Returns:
            (is_sensitive, confidence_score)
        """
        if not self.model:
            logger.warning("No model loaded, returning False")
            return False, 0.0
        
        if self.model_type == "distilbert":
            return self._predict_distilbert(log)
        elif self.model_type == "tinybert":
            return self._predict_tinybert(log)
        elif self.model_type == "sklearn":
            return self._predict_sklearn(log)
        
        return False, 0.0
    
    def _predict_distilbert(self, log: Dict[str, Any]) -> tuple[bool, float]:
        """Predict using DistilBERT zero-shot."""
        try:
            # Combine log fields into text
            text = self._log_to_text(log)
            
            # Check if using zero-shot or fine-tuned
            if hasattr(self.model, 'model') and 'zero-shot' in str(type(self.model.model)):
                # Zero-shot classification
                result = self.model(
                    text,
                    candidate_labels=[
                        "security incident",
                        "normal operation",
                        "error",
                        "attack",
                        "breach"
                    ],
                    multi_label=False
                )
                
                # Check if top label indicates sensitivity
                top_label = result['labels'][0]
                top_score = result['scores'][0]
                
                is_sensitive = top_label in ['security incident', 'attack', 'breach']
                confidence = top_score if is_sensitive else 1.0 - top_score
                
            else:
                # Fine-tuned binary classification
                result = self.model(text)[0]
                label = result['label']
                score = result['score']
                
                is_sensitive = label == 'LABEL_1' or label == 'sensitive'
                confidence = score
            
            return is_sensitive, confidence
            
        except Exception as e:
            logger.error(f"DistilBERT prediction failed: {e}")
            return False, 0.0
    
    def _predict_tinybert(self, log: Dict[str, Any]) -> tuple[bool, float]:
        """Predict using TinyBERT."""
        try:
            import torch
            
            text = self._log_to_text(log)
            
            # Tokenize
            inputs = self.tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512
            )
            
            # Predict
            with torch.no_grad():
                outputs = self.model(**inputs)
                probs = torch.softmax(outputs.logits, dim=-1)
                
                # Assume binary: [not_sensitive, sensitive]
                confidence = probs[0][1].item()
                is_sensitive = confidence >= self.confidence_threshold
            
            return is_sensitive, confidence
            
        except Exception as e:
            logger.error(f"TinyBERT prediction failed: {e}")
            return False, 0.0
    
    def _predict_sklearn(self, log: Dict[str, Any]) -> tuple[bool, float]:
        """Predict using sklearn model."""
        try:
            text = self._log_to_text(log)
            
            # Predict
            pred_proba = self.model.predict_proba([text])[0]
            confidence = pred_proba[1]  # Probability of sensitive class
            is_sensitive = confidence >= self.confidence_threshold
            
            return is_sensitive, confidence
            
        except Exception as e:
            logger.error(f"sklearn prediction failed: {e}")
            return False, 0.0
    
    def _log_to_text(self, log: Dict[str, Any]) -> str:
        """Convert log dict to text for model input."""
        level = log.get('Level', 'INFO')
        component = log.get('Component', 'unknown')
        content = log.get('Content', '')
        
        # Format: "[LEVEL] [component] content"
        return f"[{level}] [{component}] {content}"


# ============================================================================
# TRAINING SCRIPTS (Optional - only if you want to fine-tune)
# ============================================================================

def train_distilbert_on_logs(
    labeled_logs_path: str,
    output_path: str = "trained_models/sensitivity_distilbert"
):
    """
    Fine-tune DistilBERT on labeled log data.
    
    CSV format:
    Level,Component,Content,is_sensitive
    ERROR,auth,Login failed,1
    INFO,app,User logged in,0
    
    Only need 100-500 labeled examples!
    """
    try:
        from transformers import (
            AutoTokenizer,
            AutoModelForSequenceClassification,
            Trainer,
            TrainingArguments
        )
        import pandas as pd
        from datasets import Dataset
        
        # Load data
        df = pd.read_csv(labeled_logs_path)
        
        # Prepare text
        df['text'] = df.apply(
            lambda row: f"[{row['Level']}] [{row['Component']}] {row['Content']}",
            axis=1
        )
        
        # Create dataset
        dataset = Dataset.from_pandas(df[['text', 'is_sensitive']])
        dataset = dataset.rename_column('is_sensitive', 'label')
        
        # Load model
        model_name = "distilbert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            num_labels=2
        )
        
        # Tokenize
        def tokenize_function(examples):
            return tokenizer(
                examples['text'],
                padding="max_length",
                truncation=True,
                max_length=128
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_path,
            evaluation_strategy="no",
            learning_rate=2e-5,
            per_device_train_batch_size=8,
            num_train_epochs=3,
            weight_decay=0.01,
            save_strategy="epoch",
        )
        
        # Train
        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=tokenized_dataset,
        )
        
        trainer.train()
        
        # Save
        model.save_pretrained(output_path)
        tokenizer.save_pretrained(output_path)
        
        print(f"✅ Model saved to {output_path}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")


def train_sklearn_on_logs(
    labeled_logs_path: str,
    output_path: str = "trained_models/sensitivity_sklearn.pkl"
):
    """
    Train simple sklearn model (fastest option).
    
    Uses TF-IDF + Logistic Regression.
    Very fast, very small, good baseline.
    """
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.linear_model import LogisticRegression
        from sklearn.pipeline import Pipeline
        import pandas as pd
        import pickle
        
        # Load data
        df = pd.read_csv(labeled_logs_path)
        
        # Prepare text
        df['text'] = df.apply(
            lambda row: f"[{row['Level']}] [{row['Component']}] {row['Content']}",
            axis=1
        )
        
        X = df['text'].values
        y = df['is_sensitive'].values
        
        # Create pipeline
        model = Pipeline([
            ('tfidf', TfidfVectorizer(max_features=1000, ngram_range=(1, 2))),
            ('clf', LogisticRegression(max_iter=1000))
        ])
        
        # Train
        model.fit(X, y)
        
        # Save
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'wb') as f:
            pickle.dump(model, f)
        
        print(f"✅ Model saved to {output_path}")
        
        # Print accuracy
        accuracy = model.score(X, y)
        print(f"📊 Training accuracy: {accuracy:.2%}")
        
    except Exception as e:
        logger.error(f"Training failed: {e}")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example 1: Zero-shot detection (no training needed!)
    detector = MLSensitivityDetector(model_type="distilbert")
    
    test_logs = [
        {'Level': 'INFO', 'Content': 'User logged in', 'Component': 'app'},
        {'Level': 'ERROR', 'Content': 'Authentication failed', 'Component': 'auth'},
        {'Level': 'FATAL', 'Content': 'Security breach detected!', 'Component': 'security'},
    ]
    
    for log in test_logs:
        is_sens, conf = detector.predict(log)
        print(f"{log['Level']:6} | Sensitive: {is_sens:5} | Confidence: {conf:.2f} | {log['Content']}")
