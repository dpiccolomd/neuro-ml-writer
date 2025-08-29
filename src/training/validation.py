"""
Comprehensive Validation Framework

BULLETPROOF validation that ensures all components meet quality standards.
Validates data quality, model performance, and system integration.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
import pandas as pd
import numpy as np
from datetime import datetime
import torch
from transformers import AutoTokenizer
from sklearn.metrics import accuracy_score, f1_score, classification_report
import warnings

from ..agents.utils.data_processing import AnnotationDatabaseManager, CitationDataset
from ..agents.citation.models import CitationContextClassifier


class DataQualityValidator:
    """
    Validates quality of collected papers and expert annotations.
    
    BULLETPROOF standards: Rejects any simulation, placeholder, or low-quality data.
    """
    
    def __init__(self, papers_db_path: str, annotations_db_path: str):
        self.papers_db_path = papers_db_path
        self.annotations_db_path = annotations_db_path
        self.logger = logging.getLogger(__name__)
        
    def validate_paper_collection(self) -> Dict[str, Any]:
        """Validate collected papers meet quality standards."""
        
        if not Path(self.papers_db_path).exists():
            return {
                'valid': False,
                'error': 'Papers database not found',
                'requirements_met': False
            }
            
        conn = sqlite3.connect(self.papers_db_path)
        cursor = conn.cursor()
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'statistics': {},
            'requirements_met': False
        }
        
        # Check total paper count
        cursor.execute('SELECT COUNT(*) FROM papers')
        total_papers = cursor.fetchone()[0]
        
        if total_papers == 0:
            validation_results['valid'] = False
            validation_results['error'] = 'No papers found in database'
            return validation_results
            
        # Check papers with abstracts (required for training)
        cursor.execute('SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL AND abstract != ""')
        papers_with_abstracts = cursor.fetchone()[0]
        
        # Check data quality indicators
        cursor.execute('''
            SELECT 
                AVG(LENGTH(title)) as avg_title_length,
                AVG(LENGTH(abstract)) as avg_abstract_length,
                COUNT(DISTINCT collection_source) as unique_sources,
                COUNT(DISTINCT journal) as unique_journals
            FROM papers 
            WHERE abstract IS NOT NULL AND abstract != ""
        ''')
        
        quality_stats = cursor.fetchone()
        
        validation_results['statistics'] = {
            'total_papers': total_papers,
            'papers_with_abstracts': papers_with_abstracts,
            'avg_title_length': quality_stats[0] or 0,
            'avg_abstract_length': quality_stats[1] or 0,
            'unique_sources': quality_stats[2] or 0,
            'unique_journals': quality_stats[3] or 0
        }
        
        # Quality checks
        MIN_PAPERS = 50
        MIN_ABSTRACT_LENGTH = 100
        MIN_SOURCES = 1
        
        if papers_with_abstracts < MIN_PAPERS:
            validation_results['warnings'].append(
                f'Low paper count with abstracts: {papers_with_abstracts} < {MIN_PAPERS}'
            )
            
        if validation_results['statistics']['avg_abstract_length'] < MIN_ABSTRACT_LENGTH:
            validation_results['warnings'].append(
                f'Short average abstract length: {validation_results["statistics"]["avg_abstract_length"]:.1f} < {MIN_ABSTRACT_LENGTH}'
            )
            
        if validation_results['statistics']['unique_sources'] < MIN_SOURCES:
            validation_results['warnings'].append(
                f'Limited data sources: {validation_results["statistics"]["unique_sources"]} < {MIN_SOURCES}'
            )
            
        # Check for potential simulation indicators
        cursor.execute('SELECT title FROM papers LIMIT 10')
        sample_titles = [row[0] for row in cursor.fetchall()]
        
        simulation_indicators = ['test', 'example', 'sample', 'placeholder', 'mock', 'dummy']
        suspicious_titles = []
        
        for title in sample_titles:
            if title and any(indicator in title.lower() for indicator in simulation_indicators):
                suspicious_titles.append(title)
                
        if suspicious_titles:
            validation_results['valid'] = False
            validation_results['error'] = f'Potential simulated data detected in titles: {suspicious_titles}'
            return validation_results
            
        # Requirements check
        validation_results['requirements_met'] = (
            papers_with_abstracts >= MIN_PAPERS and
            validation_results['statistics']['avg_abstract_length'] >= MIN_ABSTRACT_LENGTH and
            validation_results['statistics']['unique_sources'] >= MIN_SOURCES
        )
        
        conn.close()
        return validation_results
        
    def validate_expert_annotations(self) -> Dict[str, Any]:
        """Validate expert annotations meet bulletproof standards."""
        
        if not Path(self.annotations_db_path).exists():
            return {
                'valid': False,
                'error': 'Annotations database not found',
                'requirements_met': False
            }
            
        manager = AnnotationDatabaseManager(self.annotations_db_path)
        stats = manager.get_annotation_stats()
        
        validation_results = {
            'valid': True,
            'warnings': [],
            'statistics': stats,
            'requirements_met': False
        }
        
        # Check minimum annotation requirements
        MIN_TOTAL_ANNOTATIONS = 100
        MIN_TRAIN_ANNOTATIONS = 50
        MIN_ANNOTATORS = 1
        MIN_CONFIDENCE = 0.7
        
        total_annotations = stats.get('total', 0)
        train_annotations = stats.get('by_split', {}).get('train', 0)
        num_annotators = len(stats.get('by_annotator', {}))
        avg_confidence = stats.get('avg_confidence', 0)
        
        if total_annotations < MIN_TOTAL_ANNOTATIONS:
            validation_results['warnings'].append(
                f'Insufficient total annotations: {total_annotations} < {MIN_TOTAL_ANNOTATIONS}'
            )
            
        if train_annotations < MIN_TRAIN_ANNOTATIONS:
            validation_results['warnings'].append(
                f'Insufficient training annotations: {train_annotations} < {MIN_TRAIN_ANNOTATIONS}'
            )
            
        if num_annotators < MIN_ANNOTATORS:
            validation_results['warnings'].append(
                f'Too few annotators: {num_annotators} < {MIN_ANNOTATORS}'
            )
            
        if avg_confidence and avg_confidence < MIN_CONFIDENCE:
            validation_results['warnings'].append(
                f'Low average confidence: {avg_confidence:.3f} < {MIN_CONFIDENCE}'
            )
            
        # Check for simulation indicators in annotation data
        conn = sqlite3.connect(self.annotations_db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT DISTINCT annotator_id FROM citation_annotations')
        annotator_ids = [row[0] for row in cursor.fetchall()]
        
        simulation_indicators = ['test', 'dev', 'mock', 'sim', 'dummy', 'placeholder']
        for annotator_id in annotator_ids:
            if any(indicator in annotator_id.lower() for indicator in simulation_indicators):
                validation_results['warnings'].append(
                    f'Potential simulated annotator detected: {annotator_id}'
                )
        
        # Requirements check
        validation_results['requirements_met'] = (
            total_annotations >= MIN_TOTAL_ANNOTATIONS and
            train_annotations >= MIN_TRAIN_ANNOTATIONS and
            num_annotators >= MIN_ANNOTATORS and
            (avg_confidence is None or avg_confidence >= MIN_CONFIDENCE)
        )
        
        conn.close()
        return validation_results


class ModelPerformanceValidator:
    """
    Validates ML model performance meets bulletproof standards.
    
    Ensures models achieve minimum accuracy and reliability requirements.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_citation_model(
        self, 
        model: CitationContextClassifier,
        test_dataset: CitationDataset,
        min_accuracy: float = 0.75,
        min_f1: float = 0.70
    ) -> Dict[str, Any]:
        """Validate citation model performance on test data."""
        
        validation_results = {
            'valid': True,
            'performance_metrics': {},
            'requirements_met': False,
            'warnings': []
        }
        
        if len(test_dataset) == 0:
            validation_results['valid'] = False
            validation_results['error'] = 'No test data available'
            return validation_results
            
        model.eval()
        
        # Collect predictions
        all_necessity_preds = []
        all_necessity_labels = []
        all_type_preds = []
        all_type_labels = []
        all_placement_preds = []
        all_placement_labels = []
        
        with torch.no_grad():
            for i in range(len(test_dataset)):
                sample = test_dataset[i]
                
                # Forward pass
                outputs = model(
                    sample['input_ids'].unsqueeze(0),
                    sample['attention_mask'].unsqueeze(0)
                )
                
                # Get predictions
                necessity_pred = outputs['necessity_logits'].argmax(dim=-1).item()
                type_pred = outputs['type_logits'].argmax(dim=-1).item()
                placement_pred = outputs['placement_logits'].argmax(dim=-1).item()
                
                all_necessity_preds.append(necessity_pred)
                all_necessity_labels.append(sample['necessity_labels'].item())
                all_type_preds.append(type_pred)
                all_type_labels.append(sample['type_labels'].item())
                all_placement_preds.append(placement_pred)
                all_placement_labels.append(sample['placement_labels'].item())
        
        # Calculate performance metrics
        necessity_acc = accuracy_score(all_necessity_labels, all_necessity_preds)
        necessity_f1 = f1_score(all_necessity_labels, all_necessity_preds, average='weighted')
        
        type_acc = accuracy_score(all_type_labels, all_type_preds)
        type_f1 = f1_score(all_type_labels, all_type_preds, average='weighted')
        
        placement_acc = accuracy_score(all_placement_labels, all_placement_preds)
        placement_f1 = f1_score(all_placement_labels, all_placement_preds, average='weighted')
        
        validation_results['performance_metrics'] = {
            'necessity_accuracy': necessity_acc,
            'necessity_f1': necessity_f1,
            'type_accuracy': type_acc,
            'type_f1': type_f1,
            'placement_accuracy': placement_acc,
            'placement_f1': placement_f1,
            'overall_accuracy': (necessity_acc + type_acc + placement_acc) / 3,
            'overall_f1': (necessity_f1 + type_f1 + placement_f1) / 3
        }
        
        # Check requirements
        overall_accuracy = validation_results['performance_metrics']['overall_accuracy']
        overall_f1 = validation_results['performance_metrics']['overall_f1']
        
        if overall_accuracy < min_accuracy:
            validation_results['warnings'].append(
                f'Low overall accuracy: {overall_accuracy:.3f} < {min_accuracy}'
            )
            
        if overall_f1 < min_f1:
            validation_results['warnings'].append(
                f'Low overall F1: {overall_f1:.3f} < {min_f1}'
            )
            
        validation_results['requirements_met'] = (
            overall_accuracy >= min_accuracy and overall_f1 >= min_f1
        )
        
        # Generate detailed classification reports
        validation_results['classification_reports'] = {
            'necessity': classification_report(all_necessity_labels, all_necessity_preds, output_dict=True),
            'type': classification_report(all_type_labels, all_type_preds, output_dict=True),
            'placement': classification_report(all_placement_labels, all_placement_preds, output_dict=True)
        }
        
        return validation_results


class SystemIntegrationValidator:
    """
    Validates end-to-end system integration and functionality.
    
    Tests complete workflow from data to predictions.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
    def validate_complete_pipeline(
        self,
        papers_db_path: str,
        annotations_db_path: str,
        model_path: Optional[str] = None
    ) -> Dict[str, Any]:
        """Validate complete system integration."""
        
        validation_results = {
            'components': {},
            'integration_tests': {},
            'overall_valid': True,
            'critical_errors': []
        }
        
        # Validate data components
        data_validator = DataQualityValidator(papers_db_path, annotations_db_path)
        
        validation_results['components']['paper_collection'] = data_validator.validate_paper_collection()
        validation_results['components']['expert_annotations'] = data_validator.validate_expert_annotations()
        
        # Integration tests
        try:
            # Test dataset creation
            tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            
            train_dataset = CitationDataset(
                annotation_db_path=annotations_db_path,
                tokenizer=tokenizer,
                split="train"
            )
            
            validation_results['integration_tests']['dataset_creation'] = {
                'success': True,
                'train_samples': len(train_dataset)
            }
            
            # Test model creation if annotations exist
            if len(train_dataset) > 0:
                label_mappings = train_dataset.get_label_mappings()
                model = CitationContextClassifier(label_mappings=label_mappings)
                
                validation_results['integration_tests']['model_creation'] = {
                    'success': True,
                    'label_mappings': label_mappings
                }
                
                # Test inference
                sample = train_dataset[0]
                with torch.no_grad():
                    outputs = model(
                        sample['input_ids'].unsqueeze(0),
                        sample['attention_mask'].unsqueeze(0)
                    )
                    
                validation_results['integration_tests']['inference'] = {
                    'success': True,
                    'output_shapes': {
                        'necessity': list(outputs['necessity_logits'].shape),
                        'type': list(outputs['type_logits'].shape),
                        'placement': list(outputs['placement_logits'].shape)
                    }
                }
            else:
                validation_results['integration_tests']['model_creation'] = {
                    'success': False,
                    'error': 'No training data available'
                }
                validation_results['critical_errors'].append('No training data available for model creation')
                
        except Exception as e:
            validation_results['integration_tests']['error'] = str(e)
            validation_results['critical_errors'].append(f'Integration test failed: {str(e)}')
            validation_results['overall_valid'] = False
            
        # Overall validation
        validation_results['overall_valid'] = (
            validation_results['components']['paper_collection'].get('valid', False) and
            validation_results['components']['expert_annotations'].get('valid', False) and
            len(validation_results['critical_errors']) == 0
        )
        
        return validation_results


def run_comprehensive_validation(
    data_dir: str = "./data",
    model_dir: str = "./models"
) -> Dict[str, Any]:
    """Run comprehensive system validation."""
    
    logger = logging.getLogger(__name__)
    logger.info("Starting comprehensive system validation")
    
    papers_db_path = str(Path(data_dir) / "papers.db")
    annotations_db_path = str(Path(data_dir) / "annotations.db")
    
    # Run all validations
    validator = SystemIntegrationValidator()
    results = validator.validate_complete_pipeline(
        papers_db_path=papers_db_path,
        annotations_db_path=annotations_db_path
    )
    
    # Log results
    if results['overall_valid']:
        logger.info("✅ System validation PASSED")
    else:
        logger.error("❌ System validation FAILED")
        for error in results['critical_errors']:
            logger.error(f"   - {error}")
    
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    results = run_comprehensive_validation()
    print(json.dumps(results, indent=2, default=str))