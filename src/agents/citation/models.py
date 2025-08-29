"""
Citation Intelligence Agent: Neural Models for Citation Context Classification

This module implements the core ML models for understanding when, where, and how
to place citations in scientific text with PhD-level intelligence.
"""

import torch
import torch.nn as nn
from transformers import (
    BertModel, BertTokenizer, BertForSequenceClassification,
    AutoModel, AutoTokenizer, AutoConfig
)
from typing import Dict, List, Tuple, Optional
import numpy as np


class CitationContextClassifier(nn.Module):
    """
    BERT-based model for classifying citation contexts and necessity.
    
    Trained to predict:
    - Citation necessity (binary): Does this sentence/paragraph need citations?
    - Citation type: background, method, result_support, comparison, contradiction
    - Citation placement: beginning, middle, end of sentence
    """
    
    def __init__(
        self, 
        model_name: str = "allenai/scibert_scivocab_uncased",
        label_mappings: Optional[Dict[str, Dict]] = None,  # From real expert data
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        # Load pre-trained SciBERT as base model
        self.bert = AutoModel.from_pretrained(model_name)
        self.config = AutoConfig.from_pretrained(model_name)
        
        # Configure class counts from real expert data ONLY
        if label_mappings is None:
            raise ValueError(
                "BULLETPROOF POLICY VIOLATION: label_mappings cannot be None. "
                "Model requires expert-derived label mappings from real annotation data. "
                "No hardcoded fallbacks allowed."
            )
        
        self.label_mappings = label_mappings
        
        # Validate that mappings come from expert data
        self._validate_expert_mappings()
            
        # Extract class counts from real data
        num_necessity_classes = len(self.label_mappings['necessity'])
        num_type_classes = len(self.label_mappings['type'])
        num_placement_classes = len(self.label_mappings['placement'])
        
        # Multi-task classification heads
        hidden_size = self.config.hidden_size
        
        # Citation necessity classifier
        self.necessity_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_necessity_classes)
        )
        
        # Citation type classifier (only active when citation is needed)
        self.type_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_type_classes)
        )
        
        # Citation placement classifier
        self.placement_classifier = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, 64),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(64, num_placement_classes)
        )
    
    def _validate_expert_mappings(self):
        """Validate that label mappings come from real expert data."""
        
        required_keys = ['necessity', 'type', 'placement']
        for key in required_keys:
            if key not in self.label_mappings:
                raise ValueError(f"Missing required label mapping: {key}")
            
            if not isinstance(self.label_mappings[key], dict):
                raise ValueError(f"Label mapping {key} must be a dictionary")
                
            if len(self.label_mappings[key]) == 0:
                raise ValueError(f"Label mapping {key} cannot be empty - expert data required")
        
        # Check for suspicious hardcoded patterns
        type_mapping = self.label_mappings['type']
        suspicious_patterns = [
            'background', 'method', 'support', 'comparison', 'contradiction'
        ]
        
        if (len(type_mapping) == 5 and 
            all(pattern in type_mapping for pattern in suspicious_patterns) and
            type_mapping == {pattern: i for i, pattern in enumerate(suspicious_patterns)}):
            raise ValueError(
                "SUSPICIOUS HARDCODED PATTERN DETECTED: Citation type mapping appears to be "
                "hardcoded rather than derived from expert annotations. "
                "Only expert-derived taxonomies allowed."
            )
        
    def forward(
        self, 
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        token_type_ids: Optional[torch.Tensor] = None
    ) -> Dict[str, torch.Tensor]:
        
        # Get BERT representations
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            return_dict=True
        )
        
        # Use [CLS] token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]  # [batch_size, hidden_size]
        
        # Multi-task predictions
        necessity_logits = self.necessity_classifier(cls_output)
        type_logits = self.type_classifier(cls_output)
        placement_logits = self.placement_classifier(cls_output)
        
        return {
            'necessity_logits': necessity_logits,
            'type_logits': type_logits,
            'placement_logits': placement_logits,
            'embeddings': cls_output
        }


class CitationSelectionRanker(nn.Module):
    """
    Neural model for ranking and selecting optimal citations for a given context.
    
    Given a text context and a set of candidate citations, predicts relevance scores
    and selects the most appropriate citations to include.
    """
    
    def __init__(
        self, 
        model_name: str = "allenai/scibert_scivocab_uncased",
        max_candidates: int = 50,
        embedding_dim: int = 768
    ):
        super().__init__()
        
        self.bert = AutoModel.from_pretrained(model_name)
        self.max_candidates = max_candidates
        self.embedding_dim = embedding_dim
        
        # Relevance scoring network
        self.relevance_scorer = nn.Sequential(
            nn.Linear(embedding_dim * 2, 512),  # concat text and citation embeddings
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, 1)  # relevance score
        )
        
        # Diversity penalty network (to avoid selecting too similar citations)
        self.diversity_scorer = nn.Sequential(
            nn.Linear(embedding_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )
        
    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """Encode text using BERT and return [CLS] embeddings."""
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, return_dict=True)
        return outputs.last_hidden_state[:, 0, :]  # [CLS] token
        
    def forward(
        self,
        context_input_ids: torch.Tensor,
        context_attention_mask: torch.Tensor,
        candidate_input_ids: torch.Tensor,  # [batch_size, num_candidates, seq_len]
        candidate_attention_mask: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        
        batch_size, num_candidates, seq_len = candidate_input_ids.shape
        
        # Encode context
        context_embeddings = self.encode_text(context_input_ids, context_attention_mask)
        
        # Encode candidates (reshape for batch processing)
        candidate_input_ids_flat = candidate_input_ids.view(-1, seq_len)
        candidate_attention_mask_flat = candidate_attention_mask.view(-1, seq_len)
        
        candidate_embeddings = self.encode_text(candidate_input_ids_flat, candidate_attention_mask_flat)
        candidate_embeddings = candidate_embeddings.view(batch_size, num_candidates, self.embedding_dim)
        
        # Compute relevance scores
        context_expanded = context_embeddings.unsqueeze(1).expand(-1, num_candidates, -1)
        combined_features = torch.cat([context_expanded, candidate_embeddings], dim=-1)
        
        relevance_scores = self.relevance_scorer(combined_features).squeeze(-1)  # [batch_size, num_candidates]
        
        # Compute diversity scores (penalty for similar citations)
        diversity_scores = self.diversity_scorer(candidate_embeddings).squeeze(-1)
        
        return {
            'relevance_scores': relevance_scores,
            'diversity_scores': diversity_scores,
            'context_embeddings': context_embeddings,
            'candidate_embeddings': candidate_embeddings
        }


class CitationNetworkGNN(nn.Module):
    """
    Graph Neural Network for understanding citation relationships and influence.
    
    Models the citation network as a graph and learns representations that capture
    influence, authority, and topical relationships between papers.
    """
    
    def __init__(
        self,
        input_dim: int = 768,  # BERT embedding dimension
        hidden_dim: int = 256,
        output_dim: int = 128,
        num_layers: int = 3,
        dropout_rate: float = 0.1
    ):
        super().__init__()
        
        self.num_layers = num_layers
        
        # Graph convolution layers
        self.conv_layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        # Input layer
        self.conv_layers.append(nn.Linear(input_dim, hidden_dim))
        self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
        
        # Hidden layers
        for _ in range(num_layers - 2):
            self.conv_layers.append(nn.Linear(hidden_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            
        # Output layer
        self.conv_layers.append(nn.Linear(hidden_dim, output_dim))
        
        self.dropout = nn.Dropout(dropout_rate)
        self.activation = nn.ReLU()
    
    def forward(
        self, 
        node_features: torch.Tensor,  # [num_nodes, input_dim]
        adjacency_matrix: torch.Tensor  # [num_nodes, num_nodes]
    ) -> torch.Tensor:
        
        x = node_features
        
        for i, (conv, bn) in enumerate(zip(self.conv_layers[:-1], self.batch_norms)):
            # Graph convolution: aggregate neighbor features
            x = torch.matmul(adjacency_matrix, x)  # Simple aggregation
            x = conv(x)
            x = bn(x)
            x = self.activation(x)
            x = self.dropout(x)
            
        # Final layer (no batch norm or activation)
        x = torch.matmul(adjacency_matrix, x)
        x = self.conv_layers[-1](x)
        
        return x


class IntegratedCitationAgent:
    """
    Main Citation Intelligence Agent that combines all neural models
    for comprehensive citation understanding and generation.
    """
    
    def __init__(
        self,
        context_classifier: CitationContextClassifier,
        selection_ranker: CitationSelectionRanker,
        network_gnn: CitationNetworkGNN,
        tokenizer: AutoTokenizer
    ):
        self.context_classifier = context_classifier
        self.selection_ranker = selection_ranker
        self.network_gnn = network_gnn
        self.tokenizer = tokenizer
        
        # Set models to evaluation mode
        self.context_classifier.eval()
        self.selection_ranker.eval()
        self.network_gnn.eval()
        
    def predict_citation_necessity(self, text: str) -> Dict[str, float]:
        """Predict if the given text needs citations and what type."""
        
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        with torch.no_grad():
            outputs = self.context_classifier(**inputs)
            
            # Get probabilities
            necessity_probs = torch.softmax(outputs['necessity_logits'], dim=-1)
            type_probs = torch.softmax(outputs['type_logits'], dim=-1)
            placement_probs = torch.softmax(outputs['placement_logits'], dim=-1)
            
            # Create dynamic result based on real label mappings
            result = {}
            
            # Necessity prediction (assuming binary: False=0, True=1)
            if len(necessity_probs[0]) == 2:
                result['needs_citation'] = float(necessity_probs[0, 1])
            else:
                result['needs_citation'] = float(necessity_probs[0].max())
            
            # Citation type probabilities from real data
            type_probs_dict = {}
            for type_name, type_idx in self.label_mappings['type'].items():
                if type_idx < len(type_probs[0]):
                    type_probs_dict[type_name] = float(type_probs[0, type_idx])
            result['citation_type_probs'] = type_probs_dict
            
            # Placement probabilities from real data
            placement_probs_dict = {}
            for placement_name, placement_idx in self.label_mappings['placement'].items():
                if placement_idx < len(placement_probs[0]):
                    placement_probs_dict[placement_name] = float(placement_probs[0, placement_idx])
            result['placement_probs'] = placement_probs_dict
            
            return result
    
    def select_optimal_citations(
        self, 
        context: str, 
        candidate_citations: List[str],
        max_citations: int = 3
    ) -> List[Tuple[str, float]]:
        """Select the most relevant citations for the given context."""
        
        if len(candidate_citations) == 0:
            return []
            
        # Tokenize context
        context_inputs = self.tokenizer(
            context,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Tokenize candidates
        candidate_inputs = self.tokenizer(
            candidate_citations,
            return_tensors="pt",
            max_length=512,
            truncation=True,
            padding=True
        )
        
        # Reshape for model
        batch_size = 1
        num_candidates = len(candidate_citations)
        
        candidate_input_ids = candidate_inputs['input_ids'].unsqueeze(0)
        candidate_attention_mask = candidate_inputs['attention_mask'].unsqueeze(0)
        
        with torch.no_grad():
            outputs = self.selection_ranker(
                context_inputs['input_ids'],
                context_inputs['attention_mask'],
                candidate_input_ids,
                candidate_attention_mask
            )
            
            relevance_scores = outputs['relevance_scores'][0]  # [num_candidates]
            
            # Select top citations
            _, top_indices = torch.topk(relevance_scores, min(max_citations, num_candidates))
            
            selected_citations = []
            for idx in top_indices:
                citation = candidate_citations[idx.item()]
                score = float(relevance_scores[idx.item()])
                selected_citations.append((citation, score))
                
            return selected_citations