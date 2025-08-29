"""
Data Processing Utilities for Citation Intelligence Training

BULLETPROOF data processing that connects real paper collection to training datasets.
NO simulated data, NO placeholders - only authentic expert annotations.
"""

import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from typing import Dict, List, Tuple, Optional, Any
import json
import sqlite3
from pathlib import Path
import pandas as pd
import numpy as np
from dataclasses import dataclass
import logging


@dataclass
class CitationContext:
    """
    Real citation context from expert annotations.
    
    This represents actual expert-labeled data, not simulated examples.
    """
    text: str                    # The sentence/paragraph needing citation analysis
    needs_citation: bool         # Expert annotation: Does this need citations?
    citation_type: str          # Expert annotation: background/method/support/comparison/contradiction
    citation_placement: str     # Expert annotation: beginning/middle/end
    source_paper_id: str        # Paper this context came from
    annotator_id: str          # Expert who provided this annotation
    confidence_score: float     # Annotator confidence (0.0-1.0)
    
    def __post_init__(self):
        # Validate that this is real data, not placeholder
        if not self.text.strip():
            raise ValueError("Citation context text cannot be empty - no placeholder data allowed")
        if self.citation_type not in ['background', 'method', 'support', 'comparison', 'contradiction']:
            raise ValueError(f"Invalid citation type: {self.citation_type}. Must be expert-validated category.")
        if self.citation_placement not in ['beginning', 'middle', 'end']:
            raise ValueError(f"Invalid placement: {self.citation_placement}")
        if not 0.0 <= self.confidence_score <= 1.0:
            raise ValueError("Confidence score must be between 0.0 and 1.0")


class CitationDataset(Dataset):
    """
    PyTorch Dataset for Citation Intelligence training.
    
    BULLETPROOF REQUIREMENTS:
    - Only loads real expert-annotated data from database
    - Raises exceptions if insufficient real data available  
    - No synthetic or simulated examples allowed
    """
    
    def __init__(
        self, 
        annotation_db_path: str,
        tokenizer: AutoTokenizer,
        max_length: int = 512,
        min_confidence: float = 0.8,
        split: str = "train"  # train/val/test
    ):
        self.annotation_db_path = annotation_db_path
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.min_confidence = min_confidence
        self.split = split
        
        self.logger = logging.getLogger(__name__)
        
        # Load only real expert annotations
        self.contexts = self._load_expert_annotations()
        
        if len(self.contexts) == 0:
            raise ValueError(
                f"CRITICAL: No expert annotations found in {annotation_db_path}. "
                f"Cannot train with zero real data. Expert annotation phase must be completed first."
            )
            
        self.logger.info(f"Loaded {len(self.contexts)} real expert annotations for {split} split")
        
        # Create label mappings from actual data (not hardcoded)
        self._create_label_mappings()
        
    def _load_expert_annotations(self) -> List[CitationContext]:
        """
        Load real expert annotations from database.
        
        BULLETPROOF: Only returns expert-validated data above confidence threshold.
        """
        if not Path(self.annotation_db_path).exists():
            self.logger.warning(f"Annotation database not found: {self.annotation_db_path}")
            return []
            
        try:
            conn = sqlite3.connect(self.annotation_db_path)
            
            query = """
                SELECT text, needs_citation, citation_type, citation_placement, 
                       source_paper_id, annotator_id, confidence_score, split
                FROM citation_annotations 
                WHERE confidence_score >= ? AND split = ?
                ORDER BY confidence_score DESC
            """
            
            cursor = conn.cursor()
            cursor.execute(query, (self.min_confidence, self.split))
            rows = cursor.fetchall()
            
            contexts = []
            for row in rows:
                try:
                    context = CitationContext(
                        text=row[0],
                        needs_citation=bool(row[1]),
                        citation_type=row[2],
                        citation_placement=row[3],
                        source_paper_id=row[4],
                        annotator_id=row[5],
                        confidence_score=float(row[6])
                    )
                    contexts.append(context)
                except ValueError as e:
                    self.logger.warning(f"Skipping invalid annotation: {e}")
                    continue
                    
            conn.close()
            return contexts
            
        except sqlite3.Error as e:
            self.logger.error(f"Database error loading annotations: {e}")
            return []
            
    def _create_label_mappings(self):
        """Create label mappings from actual annotation data, not hardcoded values."""
        
        # Extract unique values from real data
        necessity_values = list(set([int(ctx.needs_citation) for ctx in self.contexts]))
        type_values = list(set([ctx.citation_type for ctx in self.contexts]))
        placement_values = list(set([ctx.citation_placement for ctx in self.contexts]))
        
        # Create mappings based on actual data
        self.necessity_to_idx = {val: idx for idx, val in enumerate(sorted(necessity_values))}
        self.type_to_idx = {val: idx for idx, val in enumerate(sorted(type_values))}
        self.placement_to_idx = {val: idx for idx, val in enumerate(sorted(placement_values))}
        
        # Reverse mappings
        self.idx_to_necessity = {idx: val for val, idx in self.necessity_to_idx.items()}
        self.idx_to_type = {idx: val for val, idx in self.type_to_idx.items()}
        self.idx_to_placement = {idx: val for val, idx in self.placement_to_idx.items()}
        
        self.logger.info(f"Label mappings created from real data:")
        self.logger.info(f"  Citation types: {list(self.type_to_idx.keys())}")
        self.logger.info(f"  Placement options: {list(self.placement_to_idx.keys())}")
        
    def __len__(self) -> int:
        return len(self.contexts)
        
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        context = self.contexts[idx]
        
        # Tokenize text
        encoding = self.tokenizer(
            context.text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Convert labels to indices based on actual data
        necessity_label = self.necessity_to_idx[int(context.needs_citation)]
        type_label = self.type_to_idx[context.citation_type]
        placement_label = self.placement_to_idx[context.citation_placement]
        
        return {
            'input_ids': encoding['input_ids'].squeeze(),
            'attention_mask': encoding['attention_mask'].squeeze(),
            'necessity_labels': torch.tensor(necessity_label, dtype=torch.long),
            'type_labels': torch.tensor(type_label, dtype=torch.long),
            'placement_labels': torch.tensor(placement_label, dtype=torch.long),
            'confidence_score': torch.tensor(context.confidence_score, dtype=torch.float),
            'source_paper_id': context.source_paper_id,
            'annotator_id': context.annotator_id
        }
        
    def get_label_mappings(self) -> Dict[str, Dict]:
        """Return label mappings for model configuration."""
        return {
            'necessity': self.necessity_to_idx,
            'type': self.type_to_idx, 
            'placement': self.placement_to_idx
        }


def collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
    """
    Collate function for DataLoader.
    
    Groups real annotation samples into batches for training.
    """
    if not batch:
        raise ValueError("Cannot create batch from empty data - expert annotations required")
        
    # Stack tensors
    batch_dict = {}
    
    # Handle tensor fields
    tensor_fields = ['input_ids', 'attention_mask', 'necessity_labels', 'type_labels', 
                    'placement_labels', 'confidence_score']
    
    for field in tensor_fields:
        if field in batch[0]:
            batch_dict[field] = torch.stack([item[field] for item in batch])
    
    # Handle string fields  
    string_fields = ['source_paper_id', 'annotator_id']
    for field in string_fields:
        if field in batch[0]:
            batch_dict[field] = [item[field] for item in batch]
            
    return batch_dict


class AnnotationDatabaseManager:
    """
    Manages the expert annotation database.
    
    BULLETPROOF: Only handles real expert data, no simulation.
    """
    
    def __init__(self, db_path: str):
        self.db_path = db_path
        self.logger = logging.getLogger(__name__)
        self._init_database()
        
    def _init_database(self):
        """Initialize annotation database schema."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS citation_annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                text TEXT NOT NULL,
                needs_citation BOOLEAN NOT NULL,
                citation_type TEXT NOT NULL,
                citation_placement TEXT NOT NULL,
                source_paper_id TEXT NOT NULL,
                annotator_id TEXT NOT NULL,
                confidence_score REAL NOT NULL,
                split TEXT NOT NULL,  -- train/val/test
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                CONSTRAINT valid_citation_type CHECK (citation_type IN ('background', 'method', 'support', 'comparison', 'contradiction')),
                CONSTRAINT valid_placement CHECK (citation_placement IN ('beginning', 'middle', 'end')),
                CONSTRAINT valid_confidence CHECK (confidence_score >= 0.0 AND confidence_score <= 1.0),
                CONSTRAINT valid_split CHECK (split IN ('train', 'val', 'test'))
            )
        ''')
        
        # Index for faster queries
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_split_confidence ON citation_annotations(split, confidence_score)')
        cursor.execute('CREATE INDEX IF NOT EXISTS idx_annotator ON citation_annotations(annotator_id)')
        
        conn.commit()
        conn.close()
        
    def add_annotation(self, context: CitationContext, split: str = "train") -> bool:
        """Add expert annotation to database with validation."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''
                INSERT INTO citation_annotations 
                (text, needs_citation, citation_type, citation_placement, 
                 source_paper_id, annotator_id, confidence_score, split)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                context.text,
                context.needs_citation,
                context.citation_type,
                context.citation_placement,
                context.source_paper_id,
                context.annotator_id,
                context.confidence_score,
                split
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except sqlite3.Error as e:
            self.logger.error(f"Failed to add annotation: {e}")
            return False
            
    def get_annotation_stats(self) -> Dict[str, Any]:
        """Get statistics about expert annotations in database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total annotations by split
        cursor.execute('SELECT split, COUNT(*) FROM citation_annotations GROUP BY split')
        stats['by_split'] = dict(cursor.fetchall())
        
        # By annotator
        cursor.execute('SELECT annotator_id, COUNT(*) FROM citation_annotations GROUP BY annotator_id')
        stats['by_annotator'] = dict(cursor.fetchall())
        
        # By citation type
        cursor.execute('SELECT citation_type, COUNT(*) FROM citation_annotations GROUP BY citation_type')
        stats['by_type'] = dict(cursor.fetchall())
        
        # Average confidence
        cursor.execute('SELECT AVG(confidence_score) FROM citation_annotations')
        stats['avg_confidence'] = cursor.fetchone()[0]
        
        # Total count
        cursor.execute('SELECT COUNT(*) FROM citation_annotations')
        stats['total'] = cursor.fetchone()[0]
        
        conn.close()
        return stats


def create_train_val_test_split(
    db_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15
) -> Dict[str, int]:
    """
    Split expert annotations into train/val/test sets.
    
    BULLETPROOF: Only works with real expert data.
    """
    if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
        raise ValueError("Split ratios must sum to 1.0")
        
    logger = logging.getLogger(__name__)
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # Get all annotations without split assignment
    cursor.execute('SELECT id FROM citation_annotations WHERE split IS NULL OR split = ""')
    annotation_ids = [row[0] for row in cursor.fetchall()]
    
    if len(annotation_ids) == 0:
        logger.warning("No annotations found for splitting")
        return {'train': 0, 'val': 0, 'test': 0}
    
    # Shuffle annotations
    np.random.seed(42)  # Reproducible splits
    shuffled_ids = np.random.permutation(annotation_ids)
    
    # Calculate split sizes
    n_total = len(shuffled_ids)
    n_train = int(n_total * train_ratio)
    n_val = int(n_total * val_ratio)
    n_test = n_total - n_train - n_val
    
    # Assign splits
    train_ids = shuffled_ids[:n_train]
    val_ids = shuffled_ids[n_train:n_train + n_val]
    test_ids = shuffled_ids[n_train + n_val:]
    
    # Update database
    for split_name, ids in [('train', train_ids), ('val', val_ids), ('test', test_ids)]:
        if len(ids) > 0:
            placeholders = ','.join(['?' for _ in ids])
            cursor.execute(f'UPDATE citation_annotations SET split = ? WHERE id IN ({placeholders})', 
                         [split_name] + ids.tolist())
    
    conn.commit()
    conn.close()
    
    logger.info(f"Split {n_total} annotations: train={n_train}, val={n_val}, test={n_test}")
    
    return {'train': n_train, 'val': n_val, 'test': n_test}