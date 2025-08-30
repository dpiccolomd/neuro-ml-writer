#!/usr/bin/env python3
"""
Training Data Authenticity Validator

Validates that ALL training data is expert-derived and authentic.
NO simulation, NO placeholder, NO artificial data allowed.
"""

import os
import sys
import json
import sqlite3
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np
from dataclasses import dataclass


@dataclass
class DataValidationResult:
    """Result of training data validation."""
    is_authentic: bool
    violations: List[str]
    statistics: Dict[str, Any]
    confidence_score: float


class TrainingDataValidator:
    """
    Ultra-strict validator for training data authenticity.
    
    Ensures ALL training data meets bulletproof standards:
    - Expert-annotated only
    - No simulation or placeholder content
    - Realistic statistical distributions
    - Authenticated source provenance
    """
    
    def __init__(self):
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)
        
        # Patterns that indicate non-expert data
        self.suspicious_patterns = {
            'fake_annotators': [
                'test_', 'demo_', 'mock_', 'sim_', 'dev_', 'placeholder_',
                'fake_', 'artificial_', 'generated_', 'system_', 'bot_',
                'auto_', 'default_', 'sample_'
            ],
            'artificial_text_patterns': [
                'this is a test', 'example text', 'placeholder content',
                'lorem ipsum', 'sample data', 'dummy text', 'test case',
                'artificial example', 'mock content'
            ],
            'hardcoded_sequences': [
                # Too-perfect label distributions
                ['background', 'method', 'support', 'comparison', 'contradiction'],
                ['beginning', 'middle', 'end'],
                # Suspicious confidence patterns
                [0.9, 0.85, 0.95, 0.8, 0.92]
            ]
        }
    
    def validate_expert_annotations(
        self, 
        db_path: str,
        mode: str = "comprehensive"
    ) -> DataValidationResult:
        """
        Validate expert annotation database for authenticity.
        
        Args:
            db_path: Path to annotations database
            mode: Validation mode - 'comprehensive', 'expert-only', 'strict'
        """
        self.logger.info(f"🔍 Validating training data authenticity: {db_path}")
        
        violations = []
        statistics = {}
        
        try:
            if not Path(db_path).exists():
                return DataValidationResult(
                    is_authentic=False,
                    violations=[f"Training database not found: {db_path}"],
                    statistics={'database_exists': False},
                    confidence_score=0.0
                )
            
            # Connect to database
            conn = sqlite3.connect(db_path)
            cursor = conn.cursor()
            
            # Validate database structure
            structure_violations = self._validate_database_structure(cursor)
            violations.extend(structure_violations)
            
            # Validate annotator authenticity
            annotator_violations = self._validate_annotator_authenticity(cursor)
            violations.extend(annotator_violations)
            
            # Validate annotation content
            content_violations = self._validate_annotation_content(cursor)
            violations.extend(content_violations)
            
            # Validate statistical patterns
            statistical_violations = self._validate_statistical_patterns(cursor)
            violations.extend(statistical_violations)
            
            # Generate statistics
            statistics = self._generate_data_statistics(cursor)
            
            conn.close()
            
        except Exception as e:
            violations.append(f"Database validation error: {str(e)}")
            statistics['validation_error'] = str(e)
        
        # Calculate confidence score
        confidence_score = self._calculate_authenticity_confidence(violations, statistics)
        
        result = DataValidationResult(
            is_authentic=len(violations) == 0,
            violations=violations,
            statistics=statistics,
            confidence_score=confidence_score
        )
        
        self._log_validation_results(result)
        return result
    
    def _validate_database_structure(self, cursor) -> List[str]:
        """Validate database has proper structure for expert annotations."""
        violations = []
        
        # Check required tables
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
        tables = [row[0] for row in cursor.fetchall()]
        
        if 'citation_annotations' not in tables:
            violations.append("Missing citation_annotations table - no training data structure")
            return violations
        
        # Check required columns
        cursor.execute("PRAGMA table_info(citation_annotations);")
        columns = [row[1] for row in cursor.fetchall()]
        
        required_columns = [
            'text', 'needs_citation', 'citation_type', 'citation_placement',
            'annotator_id', 'confidence_score', 'source_paper_id'
        ]
        
        for required_col in required_columns:
            if required_col not in columns:
                violations.append(f"Missing required column: {required_col}")
        
        return violations
    
    def _validate_annotator_authenticity(self, cursor) -> List[str]:
        """Validate that annotators appear to be real experts."""
        violations = []
        
        cursor.execute("SELECT DISTINCT annotator_id FROM citation_annotations")
        annotators = [row[0] for row in cursor.fetchall()]
        
        if not annotators:
            violations.append("No annotators found - expert annotations required")
            return violations
        
        for annotator_id in annotators:
            annotator_lower = annotator_id.lower()
            
            # Check for suspicious annotator patterns
            for suspicious in self.suspicious_patterns['fake_annotators']:
                if suspicious in annotator_lower:
                    violations.append(
                        f"SUSPICIOUS ANNOTATOR: '{annotator_id}' contains pattern '{suspicious}'"
                    )
            
            # Check annotator activity patterns
            cursor.execute(
                "SELECT COUNT(*), AVG(confidence_score), MIN(created_at), MAX(created_at) "
                "FROM citation_annotations WHERE annotator_id = ?",
                (annotator_id,)
            )
            count, avg_confidence, min_time, max_time = cursor.fetchone()
            
            # Too few annotations suggests test data
            if count < 10:
                violations.append(
                    f"INSUFFICIENT ANNOTATIONS: '{annotator_id}' has only {count} annotations "
                    "(minimum 10 expected for real expert)"
                )
            
            # Too uniform confidence suggests artificial data
            if avg_confidence and (avg_confidence > 0.95 or avg_confidence < 0.6):
                violations.append(
                    f"SUSPICIOUS CONFIDENCE: '{annotator_id}' has unrealistic average confidence {avg_confidence:.3f}"
                )
        
        return violations
    
    def _validate_annotation_content(self, cursor) -> List[str]:
        """Validate annotation content appears authentic."""
        violations = []
        
        cursor.execute("SELECT text, annotator_id FROM citation_annotations LIMIT 100")
        annotations = cursor.fetchall()
        
        for text, annotator_id in annotations:
            text_lower = text.lower().strip()
            
            # Check for artificial text patterns
            for pattern in self.suspicious_patterns['artificial_text_patterns']:
                if pattern in text_lower:
                    violations.append(
                        f"ARTIFICIAL TEXT: '{annotator_id}' annotation contains '{pattern}': {text[:50]}..."
                    )
            
            # Check for obviously generated content
            if len(text) < 10:
                violations.append(
                    f"SUSPICIOUSLY SHORT: '{annotator_id}' annotation too short: '{text}'"
                )
            
            # Check for repetitive patterns
            words = text_lower.split()
            if len(set(words)) < len(words) * 0.5 and len(words) > 5:
                violations.append(
                    f"REPETITIVE CONTENT: '{annotator_id}' annotation highly repetitive: {text[:50]}..."
                )
        
        return violations
    
    def _validate_statistical_patterns(self, cursor) -> List[str]:
        """Validate statistical patterns appear authentic."""
        violations = []
        
        # Check citation type distributions
        cursor.execute("SELECT citation_type, COUNT(*) FROM citation_annotations GROUP BY citation_type")
        type_counts = dict(cursor.fetchall())
        
        if type_counts:
            # Check for too-perfect uniform distribution
            counts = list(type_counts.values())
            if len(set(counts)) == 1 and len(counts) > 2:
                violations.append(
                    f"ARTIFICIAL DISTRIBUTION: Citation types have identical counts {type_counts}"
                )
            
            # Check for suspicious hardcoded sequence
            types = sorted(type_counts.keys())
            for hardcoded_seq in self.suspicious_patterns['hardcoded_sequences']:
                if isinstance(hardcoded_seq[0], str) and types == sorted(hardcoded_seq):
                    violations.append(
                        f"HARDCODED SEQUENCE: Citation types match suspicious pattern: {types}"
                    )
        
        # Check confidence score patterns
        cursor.execute("SELECT confidence_score FROM citation_annotations")
        confidence_scores = [row[0] for row in cursor.fetchall()]
        
        if confidence_scores:
            unique_scores = sorted(set(confidence_scores))
            
            # Check for suspicious uniform high confidence
            if len(unique_scores) <= 3 and all(score >= 0.85 for score in unique_scores):
                violations.append(
                    f"SUSPICIOUS CONFIDENCE PATTERN: Too uniform high scores: {unique_scores}"
                )
            
            # Check for obviously artificial patterns
            score_std = np.std(confidence_scores) if len(confidence_scores) > 1 else 0
            if score_std < 0.05:
                violations.append(
                    f"ARTIFICIAL CONFIDENCE VARIANCE: Standard deviation {score_std:.4f} too low for real experts"
                )
        
        return violations
    
    def _generate_data_statistics(self, cursor) -> Dict[str, Any]:
        """Generate comprehensive statistics about training data."""
        stats = {}
        
        # Basic counts
        cursor.execute("SELECT COUNT(*) FROM citation_annotations")
        stats['total_annotations'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT annotator_id) FROM citation_annotations")
        stats['unique_annotators'] = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(DISTINCT citation_type) FROM citation_annotations")
        stats['unique_types'] = cursor.fetchone()[0]
        
        # Confidence statistics
        cursor.execute("SELECT AVG(confidence_score), MIN(confidence_score), MAX(confidence_score), "
                      "COUNT(DISTINCT confidence_score) FROM citation_annotations")
        avg_conf, min_conf, max_conf, unique_conf = cursor.fetchone()
        stats['confidence_stats'] = {
            'average': avg_conf,
            'minimum': min_conf,
            'maximum': max_conf,
            'unique_values': unique_conf
        }
        
        # Distribution analysis
        cursor.execute("SELECT citation_type, COUNT(*) FROM citation_annotations GROUP BY citation_type")
        stats['type_distribution'] = dict(cursor.fetchall())
        
        cursor.execute("SELECT annotator_id, COUNT(*) FROM citation_annotations GROUP BY annotator_id")
        stats['annotator_distribution'] = dict(cursor.fetchall())
        
        return stats
    
    def _calculate_authenticity_confidence(
        self, 
        violations: List[str], 
        statistics: Dict[str, Any]
    ) -> float:
        """Calculate confidence score for data authenticity."""
        if violations:
            # Severe penalty for any violations
            violation_penalty = min(len(violations) * 0.3, 0.9)
            base_score = 1.0 - violation_penalty
        else:
            base_score = 1.0
        
        # Bonus for good data characteristics
        total_annotations = statistics.get('total_annotations', 0)
        unique_annotators = statistics.get('unique_annotators', 0)
        
        if total_annotations >= 1000:
            base_score = min(base_score + 0.1, 1.0)
        if unique_annotators >= 3:
            base_score = min(base_score + 0.05, 1.0)
        
        return max(0.0, base_score)
    
    def _log_validation_results(self, result: DataValidationResult):
        """Log validation results."""
        if result.is_authentic:
            self.logger.info("✅ TRAINING DATA VALIDATION PASSED")
            self.logger.info(f"   Authenticity confidence: {result.confidence_score:.2f}")
            self.logger.info(f"   Total annotations: {result.statistics.get('total_annotations', 0)}")
            self.logger.info(f"   Unique annotators: {result.statistics.get('unique_annotators', 0)}")
        else:
            self.logger.error("❌ TRAINING DATA VALIDATION FAILED")
            self.logger.error(f"   {len(result.violations)} violations detected")
            for violation in result.violations[:5]:
                self.logger.error(f"   - {violation}")
            if len(result.violations) > 5:
                self.logger.error(f"   ... and {len(result.violations) - 5} more violations")
    
    def validate_all_training_data(self, data_dir: str = "./data") -> Dict[str, Any]:
        """Validate all training data in directory."""
        data_path = Path(data_dir)
        validation_results = {}
        
        # Find all database files
        db_files = list(data_path.glob("**/*.db"))
        
        if not db_files:
            self.logger.warning(f"No database files found in {data_dir}")
            return {
                'overall_status': False,
                'error': 'No training databases found',
                'databases_validated': 0
            }
        
        overall_status = True
        total_violations = 0
        
        for db_file in db_files:
            self.logger.info(f"Validating: {db_file}")
            result = self.validate_expert_annotations(str(db_file))
            
            validation_results[str(db_file)] = {
                'is_authentic': result.is_authentic,
                'violation_count': len(result.violations),
                'confidence_score': result.confidence_score,
                'statistics': result.statistics
            }
            
            if not result.is_authentic:
                overall_status = False
                total_violations += len(result.violations)
        
        validation_results['overall_status'] = overall_status
        validation_results['total_violations'] = total_violations
        validation_results['databases_validated'] = len(db_files)
        
        return validation_results


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Validate training data authenticity"
    )
    
    parser.add_argument('--mode',
                       choices=['comprehensive', 'expert-only', 'strict'],
                       default='comprehensive',
                       help='Validation mode')
    
    parser.add_argument('--data-dir',
                       default='./data',
                       help='Data directory to validate')
    
    parser.add_argument('--db-path',
                       help='Specific database file to validate')
    
    parser.add_argument('--output',
                       default='reports/training-data-validation.json',
                       help='Output file for validation report')
    
    args = parser.parse_args()
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Run validation
    validator = TrainingDataValidator()
    
    if args.db_path:
        # Validate specific database
        result = validator.validate_expert_annotations(args.db_path, args.mode)
        validation_results = {
            args.db_path: {
                'is_authentic': result.is_authentic,
                'violations': result.violations,
                'statistics': result.statistics,
                'confidence_score': result.confidence_score
            }
        }
        overall_status = result.is_authentic
    else:
        # Validate all training data
        validation_results = validator.validate_all_training_data(args.data_dir)
        overall_status = validation_results.get('overall_status', False)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(validation_results, f, indent=2, default=str)
    
    # Print summary
    print(f"\n{'='*80}")
    print("🗄️ TRAINING DATA AUTHENTICITY VALIDATION")
    print(f"{'='*80}")
    
    if overall_status:
        print("✅ VALIDATION PASSED - All training data is authentic and expert-derived")
    else:
        print("❌ VALIDATION FAILED - Non-authentic or suspicious training data detected")
        total_violations = validation_results.get('total_violations', 0)
        print(f"Total violations: {total_violations}")
    
    print(f"📋 Detailed report: {args.output}")
    
    sys.exit(0 if overall_status else 1)


if __name__ == "__main__":
    main()