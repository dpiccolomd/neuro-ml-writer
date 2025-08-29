"""
Bulletproof Content Validator

ULTRA-STRICT validation that rejects ANY simulated, placeholder, or non-expert content.
This is the last line of defense against policy violations.
"""

import sqlite3
import json
import re
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from datetime import datetime


class BulletproofValidator:
    """
    BULLETPROOF validator that enforces zero-simulation policy.
    
    Rejects ANY content that appears to be simulated, placeholder, or non-expert.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Patterns that indicate simulation or placeholder content
        self.forbidden_patterns = {
            'simulation_indicators': [
                'simulate', 'simulation', 'simulated', 'mock', 'fake', 'artificial',
                'generated', 'placeholder', 'example', 'sample', 'dummy', 'test',
                'dev_', 'development', 'prototype', 'demo'
            ],
            'suspicious_annotators': [
                'dev_annotator', 'test_annotator', 'mock_expert', 'sim_expert',
                'demo_user', 'placeholder_expert', 'fake_annotator', 'system_generated'
            ],
            'hardcoded_sequences': [
                # Common hardcoded citation type sequences
                ['background', 'method', 'support', 'comparison', 'contradiction'],
                # Common hardcoded placement sequences  
                ['beginning', 'middle', 'end'],
                # Suspicious confidence patterns
                [0.9, 0.85, 0.95]  # Too uniform to be real expert data
            ]
        }
        
    def validate_annotation_database(self, db_path: str) -> Dict[str, Any]:
        """
        BULLETPROOF validation of annotation database.
        
        Rejects any database containing simulated or suspicious content.
        """
        
        validation_result = {
            'is_bulletproof': True,
            'violations': [],
            'warnings': [],
            'statistics': {},
            'expert_validation': {}
        }
        
        if not Path(db_path).exists():
            validation_result['is_bulletproof'] = False
            validation_result['violations'].append('Annotation database does not exist')
            return validation_result
            
        try:
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Check database schema
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if 'citation_annotations' not in tables:
                validation_result['is_bulletproof'] = False
                validation_result['violations'].append('Missing citation_annotations table')
                return validation_result
            
            # Get all annotations for analysis
            cursor.execute('''
                SELECT annotator_id, text, citation_type, citation_placement, 
                       confidence_score, created_at
                FROM citation_annotations
            ''')
            
            annotations = cursor.fetchall()
            
            if not annotations:
                validation_result['is_bulletproof'] = False
                validation_result['violations'].append('No annotations found - expert data required')
                return validation_result
            
            # Statistical analysis
            validation_result['statistics'] = {
                'total_annotations': len(annotations),
                'unique_annotators': len(set([row[0] for row in annotations])),
                'unique_citation_types': len(set([row[2] for row in annotations])),
                'avg_confidence': sum([row[4] for row in annotations]) / len(annotations)
            }
            
            # Validate annotators
            annotator_violations = self._validate_annotators(annotations)
            validation_result['violations'].extend(annotator_violations)
            
            # Validate content patterns
            content_violations = self._validate_content_patterns(annotations)
            validation_result['violations'].extend(content_violations)
            
            # Validate statistical patterns
            statistical_violations = self._validate_statistical_patterns(annotations)
            validation_result['violations'].extend(statistical_violations)
            
            # Expert validation checks
            expert_validation = self._validate_expert_characteristics(annotations)
            validation_result['expert_validation'] = expert_validation
            validation_result['violations'].extend(expert_validation.get('violations', []))
            
            conn.close()
            
        except Exception as e:
            validation_result['is_bulletproof'] = False
            validation_result['violations'].append(f'Database validation error: {str(e)}')
            
        # Final determination
        validation_result['is_bulletproof'] = len(validation_result['violations']) == 0
        
        if not validation_result['is_bulletproof']:
            self.logger.error("BULLETPROOF VALIDATION FAILED")
            for violation in validation_result['violations']:
                self.logger.error(f"  - {violation}")
        else:
            self.logger.info("✅ BULLETPROOF VALIDATION PASSED")
            
        return validation_result
        
    def _validate_annotators(self, annotations: List[Tuple]) -> List[str]:
        """Validate that annotators appear to be real experts."""
        
        violations = []
        annotator_ids = [row[0] for row in annotations]
        
        for annotator_id in set(annotator_ids):
            annotator_lower = annotator_id.lower()
            
            # Check for suspicious annotator names
            for suspicious in self.forbidden_patterns['suspicious_annotators']:
                if suspicious in annotator_lower:
                    violations.append(
                        f"SUSPICIOUS ANNOTATOR: '{annotator_id}' contains forbidden pattern '{suspicious}'"
                    )
            
            # Check for simulation indicators in annotator names
            for indicator in self.forbidden_patterns['simulation_indicators']:
                if indicator in annotator_lower:
                    violations.append(
                        f"SIMULATION INDICATOR: Annotator '{annotator_id}' contains '{indicator}'"
                    )
                    
        return violations
        
    def _validate_content_patterns(self, annotations: List[Tuple]) -> List[str]:
        """Validate that annotation content appears genuine."""
        
        violations = []
        
        for annotation in annotations:
            text = annotation[1]  # text field
            
            # Check for simulation indicators in text
            text_lower = text.lower()
            for indicator in self.forbidden_patterns['simulation_indicators']:
                if indicator in text_lower:
                    violations.append(
                        f"SIMULATION CONTENT: Text contains forbidden pattern '{indicator}': {text[:100]}..."
                    )
            
            # Check for obviously artificial patterns
            if re.match(r'^(Previous studies|The experimental procedure|Our results)', text):
                violations.append(
                    f"HARDCODED PATTERN: Text appears to be hardcoded example: {text[:50]}..."
                )
                
        return violations
        
    def _validate_statistical_patterns(self, annotations: List[Tuple]) -> List[str]:
        """Validate that statistical patterns appear genuine."""
        
        violations = []
        
        # Extract citation types and check for hardcoded sequences
        citation_types = [row[2] for row in annotations]
        unique_types = list(set(citation_types))
        unique_types.sort()
        
        # Check against known hardcoded sequences
        for hardcoded_seq in self.forbidden_patterns['hardcoded_sequences']:
            if isinstance(hardcoded_seq[0], str) and unique_types == sorted(hardcoded_seq):
                violations.append(
                    f"HARDCODED SEQUENCE: Citation types match suspicious pattern: {unique_types}"
                )
        
        # Check confidence score patterns
        confidence_scores = [row[4] for row in annotations]
        unique_confidences = sorted(list(set(confidence_scores)))
        
        # Too uniform confidence scores are suspicious
        if len(unique_confidences) <= 3 and all(c >= 0.8 for c in unique_confidences):
            violations.append(
                f"SUSPICIOUS CONFIDENCE PATTERN: Too uniform high confidence scores: {unique_confidences}"
            )
        
        # Check for artificially perfect distributions
        type_counts = {}
        for ctype in citation_types:
            type_counts[ctype] = type_counts.get(ctype, 0) + 1
            
        # All types having exactly equal counts is suspicious
        counts = list(type_counts.values())
        if len(set(counts)) == 1 and len(counts) > 2:
            violations.append(
                f"ARTIFICIAL DISTRIBUTION: All citation types have identical counts: {type_counts}"
            )
            
        return violations
        
    def _validate_expert_characteristics(self, annotations: List[Tuple]) -> Dict[str, Any]:
        """Validate characteristics that indicate real expert annotations."""
        
        expert_validation = {
            'violations': [],
            'expert_indicators': {},
            'authenticity_score': 0.0
        }
        
        # Real expert annotations should show:
        # 1. Some disagreement/variation in confidence scores
        # 2. Realistic annotation timing patterns
        # 3. Variation in text complexity and length
        
        confidence_scores = [row[4] for row in annotations]
        confidence_std = self._calculate_std(confidence_scores)
        
        if confidence_std < 0.05:  # Too little variation
            expert_validation['violations'].append(
                f"UNREALISTIC CONFIDENCE VARIATION: std={confidence_std:.3f} (too uniform for real experts)"
            )
            
        # Check annotation timing (if available)
        timestamps = [row[5] for row in annotations if row[5]]  # created_at
        if timestamps:
            # Real annotations should span reasonable time period
            timestamps_parsed = []
            for ts in timestamps:
                try:
                    dt = datetime.fromisoformat(ts)
                    timestamps_parsed.append(dt)
                except:
                    continue
                    
            if len(timestamps_parsed) > 1:
                time_span = max(timestamps_parsed) - min(timestamps_parsed)
                if time_span.total_seconds() < 60:  # All within 1 minute is suspicious
                    expert_validation['violations'].append(
                        "SUSPICIOUS TIMING: All annotations created within 1 minute (likely automated)"
                    )
        
        # Calculate overall authenticity score
        violations_count = len(expert_validation['violations'])
        expert_validation['authenticity_score'] = max(0.0, 1.0 - (violations_count * 0.3))
        
        return expert_validation
        
    def _calculate_std(self, values: List[float]) -> float:
        """Calculate standard deviation."""
        if not values:
            return 0.0
        mean = sum(values) / len(values)
        variance = sum((x - mean) ** 2 for x in values) / len(values)
        return variance ** 0.5
        
    def validate_model_training_data(
        self, 
        label_mappings: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Validate that model training data is expert-derived."""
        
        validation_result = {
            'is_bulletproof': True,
            'violations': [],
            'label_analysis': {}
        }
        
        # Check for hardcoded label patterns
        if 'type' in label_mappings:
            type_mapping = label_mappings['type']
            
            # Check for the exact hardcoded pattern we forbid
            expected_hardcoded = {
                'background': 0, 'method': 1, 'support': 2, 
                'comparison': 3, 'contradiction': 4
            }
            
            if type_mapping == expected_hardcoded:
                validation_result['is_bulletproof'] = False
                validation_result['violations'].append(
                    "HARDCODED LABEL MAPPING DETECTED: Citation type mapping is exactly the "
                    "hardcoded pattern, not derived from expert data"
                )
        
        # Analyze label distribution for artificial patterns
        for category, mapping in label_mappings.items():
            if len(mapping) > 1:
                # Check if indices are sequential starting from 0
                indices = sorted(mapping.values())
                expected_sequential = list(range(len(mapping)))
                
                if indices == expected_sequential:
                    # Sequential is normal, but combined with other factors might be suspicious
                    validation_result['label_analysis'][f'{category}_sequential'] = True
                    
        return validation_result
        
    def validate_complete_system(
        self, 
        papers_db_path: str,
        annotations_db_path: str,
        label_mappings: Optional[Dict[str, Dict]] = None
    ) -> Dict[str, Any]:
        """
        Complete bulletproof validation of the entire system.
        
        This is the final check before allowing any training to proceed.
        """
        
        complete_validation = {
            'is_completely_bulletproof': True,
            'component_validations': {},
            'overall_violations': [],
            'recommendation': ''
        }
        
        # Validate annotation database
        if Path(annotations_db_path).exists():
            db_validation = self.validate_annotation_database(annotations_db_path)
            complete_validation['component_validations']['annotation_database'] = db_validation
            
            if not db_validation['is_bulletproof']:
                complete_validation['overall_violations'].extend(
                    [f"DB: {v}" for v in db_validation['violations']]
                )
        else:
            complete_validation['overall_violations'].append(
                "No annotation database found - expert annotations required"
            )
        
        # Validate model training data if provided
        if label_mappings:
            model_validation = self.validate_model_training_data(label_mappings)
            complete_validation['component_validations']['model_data'] = model_validation
            
            if not model_validation['is_bulletproof']:
                complete_validation['overall_violations'].extend(
                    [f"MODEL: {v}" for v in model_validation['violations']]
                )
        
        # Final determination
        complete_validation['is_completely_bulletproof'] = len(complete_validation['overall_violations']) == 0
        
        # Recommendation
        if complete_validation['is_completely_bulletproof']:
            complete_validation['recommendation'] = "✅ SYSTEM IS BULLETPROOF - Training may proceed"
        else:
            complete_validation['recommendation'] = (
                "❌ SYSTEM NOT BULLETPROOF - Must fix violations before training:\n" +
                "\n".join([f"  • {v}" for v in complete_validation['overall_violations']])
            )
        
        return complete_validation


def run_bulletproof_validation(
    papers_db_path: str = "./data/papers.db",
    annotations_db_path: str = "./data/annotations.db"
) -> bool:
    """
    Run complete bulletproof validation and return True if system is safe.
    
    This should be called before ANY training begins.
    """
    
    validator = BulletproofValidator()
    result = validator.validate_complete_system(
        papers_db_path=papers_db_path,
        annotations_db_path=annotations_db_path
    )
    
    print("="*80)
    print("BULLETPROOF SYSTEM VALIDATION")
    print("="*80)
    print(result['recommendation'])
    print("="*80)
    
    return result['is_completely_bulletproof']


if __name__ == "__main__":
    import sys
    
    logging.basicConfig(level=logging.INFO)
    
    is_bulletproof = run_bulletproof_validation()
    
    if not is_bulletproof:
        print("\n🚨 SYSTEM FAILED BULLETPROOF VALIDATION")
        print("Training cannot proceed until all violations are fixed.")
        sys.exit(1)
    else:
        print("\n✅ SYSTEM PASSED BULLETPROOF VALIDATION")
        print("System meets all bulletproof standards.")
        sys.exit(0)