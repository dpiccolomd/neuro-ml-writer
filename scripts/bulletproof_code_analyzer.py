#!/usr/bin/env python3
"""
Bulletproof Code Analyzer

Ultra-comprehensive static and dynamic analysis agent that detects ANY
simulation, placeholder, or non-expert content in the codebase.

This is the automated enforcement of bulletproof standards.
"""

import ast
import os
import re
import sys
import json
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass
from datetime import datetime
import sqlite3


@dataclass
class Violation:
    """Represents a bulletproof policy violation."""
    type: str           # Type of violation
    severity: str       # critical/high/medium/low
    file: str          # File path
    line: int          # Line number
    column: int        # Column number
    description: str   # Human-readable description
    evidence: str      # Code snippet or evidence
    suggestion: str    # How to fix


class BulletproofCodeAnalyzer:
    """
    Comprehensive code analyzer for detecting bulletproof violations.
    
    Detects:
    - Simulation/placeholder patterns
    - Hardcoded values and fallbacks
    - Mock/fake/test content in production
    - Suspicious ML model architectures
    - Non-expert training data
    """
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.violations: List[Violation] = []
        
        # Setup logging
        logging.basicConfig(level=logging.INFO, 
                          format='%(asctime)s - %(levelname)s - %(message)s')
        self.logger = logging.getLogger(__name__)
        
        # Forbidden patterns - expanded and comprehensive
        self.forbidden_patterns = {
            # Simulation indicators
            'simulation_keywords': [
                'simulate', 'simulation', 'simulated', 'mock', 'fake', 'artificial',
                'generated', 'placeholder', 'example', 'sample', 'dummy', 'test_data',
                'dev_', 'development', 'prototype', 'demo', 'stub', 'hardcoded_test',
                'mock_expert', 'fake_annotation', 'sim_', 'temp_', 'todo_implement'
            ],
            
            # Suspicious function names
            'forbidden_functions': [
                'simulate_expert_annotations', 'create_fake_data', 'generate_mock_',
                'hardcode_labels', 'create_placeholder_', 'mock_database_',
                'fake_expert_', 'demo_data_', 'test_annotation_', 'stub_implementation'
            ],
            
            # Suspicious variable patterns
            'suspicious_variables': [
                'use_simulation', 'allow_fake', 'enable_mock', 'demo_mode',
                'test_mode', 'placeholder_data', 'hardcoded_', 'mock_response',
                'fake_confidence', 'artificial_score'
            ],
            
            # Hardcoded data patterns (regex)
            'hardcoded_data_patterns': [
                r'\[0\.\d+,\s*0\.\d+,\s*0\.\d+\]',  # Hardcoded confidence arrays
                r'{\s*[\'"]background[\'"]\s*:\s*0\s*,\s*[\'"]method[\'"]\s*:\s*1',  # Hardcoded label mappings
                r'[\'"]test_annotator[\'"]|[\'"]mock_expert[\'"]|[\'"]demo_user[\'"]',  # Fake annotator names
            ],
            
            # Suspicious imports
            'forbidden_imports': [
                'unittest.mock', 'pytest.mock', 'mock', 'faker', 'factory_boy'
            ]
        }
        
    def analyze_codebase(self, mode: str = "comprehensive") -> Dict[str, Any]:
        """
        Comprehensive analysis of entire codebase for bulletproof violations.
        
        Args:
            mode: Analysis mode - 'static', 'database', 'advanced', 'comprehensive'
        """
        self.logger.info(f"🔍 Starting bulletproof analysis in {mode} mode")
        
        results = {
            'analysis_mode': mode,
            'timestamp': datetime.now().isoformat(),
            'violations': [],
            'summary': {},
            'bulletproof_status': True
        }
        
        try:
            if mode in ['static', 'comprehensive']:
                self._analyze_static_patterns()
                
            if mode in ['database', 'comprehensive']:
                self._analyze_database_content()
                
            if mode in ['advanced', 'comprehensive']:
                self._analyze_ml_patterns()
                self._analyze_ast_patterns()
                
            # Compile results
            results['violations'] = [self._violation_to_dict(v) for v in self.violations]
            results['summary'] = self._generate_summary()
            results['bulletproof_status'] = len(self.violations) == 0
            
            if results['bulletproof_status']:
                self.logger.info("✅ BULLETPROOF VALIDATION PASSED - No violations detected")
            else:
                self.logger.error(f"❌ BULLETPROOF VALIDATION FAILED - {len(self.violations)} violations detected")
                
        except Exception as e:
            self.logger.error(f"Analysis failed: {str(e)}")
            results['error'] = str(e)
            results['bulletproof_status'] = False
            
        return results
    
    def _analyze_static_patterns(self):
        """Static analysis for simulation/placeholder patterns."""
        self.logger.info("🔍 Running static pattern analysis...")
        
        # Analyze Python files
        python_files = list(self.project_root.glob("**/*.py"))
        
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    lines = content.split('\n')
                    
                self._check_forbidden_patterns(file_path, content, lines)
                self._check_hardcoded_data(file_path, content, lines)
                self._check_suspicious_imports(file_path, content)
                
            except Exception as e:
                self.logger.warning(f"Could not analyze {file_path}: {e}")
                
    def _analyze_database_content(self):
        """Analyze database files for simulated content."""
        self.logger.info("🗄️ Analyzing database content...")
        
        # Check for database files
        db_files = list(self.project_root.glob("**/*.db"))
        
        for db_file in db_files:
            self._validate_database_authenticity(db_file)
            
    def _analyze_ml_patterns(self):
        """Advanced ML-specific pattern analysis."""
        self.logger.info("🧠 Analyzing ML model patterns...")
        
        # Check model files for suspicious patterns
        model_files = list(self.project_root.glob("**/models.py")) + \
                     list(self.project_root.glob("**/model*.py"))
        
        for model_file in model_files:
            self._check_ml_model_violations(model_file)
            
    def _analyze_ast_patterns(self):
        """Advanced AST analysis for suspicious code structures."""
        self.logger.info("🌳 Running AST analysis...")
        
        python_files = list(self.project_root.glob("**/*.py"))
        
        for file_path in python_files:
            if self._should_skip_file(file_path):
                continue
                
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                    
                tree = ast.parse(content, filename=str(file_path))
                self._analyze_ast_tree(tree, file_path, content.split('\n'))
                
            except SyntaxError:
                self.logger.warning(f"Syntax error in {file_path}, skipping AST analysis")
            except Exception as e:
                self.logger.warning(f"Could not parse AST for {file_path}: {e}")
                
    def _check_forbidden_patterns(self, file_path: Path, content: str, lines: List[str]):
        """Check for forbidden simulation patterns."""
        content_lower = content.lower()
        
        # Skip validation scripts themselves (they legitimately contain patterns to detect)
        if self._is_validation_script(file_path):
            return
        
        # Check simulation keywords
        for keyword in self.forbidden_patterns['simulation_keywords']:
            if keyword in content_lower:
                for line_num, line in enumerate(lines, 1):
                    if keyword in line.lower():
                        # Skip lines that are clearly detecting/rejecting these patterns
                        if self._is_legitimate_pattern_usage(line):
                            continue
                            
                        self.violations.append(Violation(
                            type="SIMULATION_KEYWORD",
                            severity="critical",
                            file=str(file_path.relative_to(self.project_root)),
                            line=line_num,
                            column=line.lower().find(keyword),
                            description=f"Forbidden simulation keyword '{keyword}' detected",
                            evidence=line.strip(),
                            suggestion=f"Remove all references to '{keyword}' and replace with real implementation"
                        ))
        
        # Check forbidden functions
        for func_name in self.forbidden_patterns['forbidden_functions']:
            if func_name in content:
                for line_num, line in enumerate(lines, 1):
                    if func_name in line:
                        self.violations.append(Violation(
                            type="FORBIDDEN_FUNCTION",
                            severity="critical",
                            file=str(file_path.relative_to(self.project_root)),
                            line=line_num,
                            column=line.find(func_name),
                            description=f"Forbidden function '{func_name}' detected",
                            evidence=line.strip(),
                            suggestion=f"Remove function '{func_name}' and implement with real expert data"
                        ))
    
    def _check_hardcoded_data(self, file_path: Path, content: str, lines: List[str]):
        """Check for hardcoded data patterns."""
        # Skip validation scripts
        if self._is_validation_script(file_path):
            return
            
        for pattern in self.forbidden_patterns['hardcoded_data_patterns']:
            matches = re.finditer(pattern, content, re.IGNORECASE | re.MULTILINE)
            
            for match in matches:
                # Find line number
                line_num = content[:match.start()].count('\n') + 1
                line_content = lines[line_num - 1] if line_num <= len(lines) else ""
                
                # Skip legitimate validation contexts
                if self._is_legitimate_pattern_usage(line_content):
                    continue
                
                self.violations.append(Violation(
                    type="HARDCODED_DATA",
                    severity="high",
                    file=str(file_path.relative_to(self.project_root)),
                    line=line_num,
                    column=match.start() - content.rfind('\n', 0, match.start()) - 1,
                    description="Hardcoded data pattern detected (likely simulation)",
                    evidence=match.group().strip(),
                    suggestion="Replace hardcoded data with expert-derived values"
                ))
    
    def _check_suspicious_imports(self, file_path: Path, content: str):
        """Check for forbidden imports."""
        lines = content.split('\n')
        
        for line_num, line in enumerate(lines, 1):
            for forbidden_import in self.forbidden_patterns['forbidden_imports']:
                if f"import {forbidden_import}" in line or f"from {forbidden_import}" in line:
                    self.violations.append(Violation(
                        type="FORBIDDEN_IMPORT",
                        severity="high",
                        file=str(file_path.relative_to(self.project_root)),
                        line=line_num,
                        column=0,
                        description=f"Forbidden import '{forbidden_import}' detected",
                        evidence=line.strip(),
                        suggestion="Remove mock/test imports from production code"
                    ))
    
    def _validate_database_authenticity(self, db_file: Path):
        """Validate database contains only authentic expert data."""
        try:
            conn = sqlite3.connect(str(db_file))
            cursor = conn.cursor()
            
            # Check if it's an annotation database
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            
            if 'citation_annotations' in tables:
                # Validate annotation authenticity
                cursor.execute("SELECT annotator_id, COUNT(*) FROM citation_annotations GROUP BY annotator_id")
                annotators = cursor.fetchall()
                
                for annotator_id, count in annotators:
                    for suspicious in self.forbidden_patterns['simulation_keywords']:
                        if suspicious in annotator_id.lower():
                            self.violations.append(Violation(
                                type="FAKE_ANNOTATOR",
                                severity="critical",
                                file=str(db_file.relative_to(self.project_root)),
                                line=0,
                                column=0,
                                description=f"Suspicious annotator ID detected: '{annotator_id}'",
                                evidence=f"Annotator '{annotator_id}' has {count} annotations",
                                suggestion="Replace with real expert annotator IDs"
                            ))
                            
            conn.close()
            
        except Exception as e:
            self.logger.warning(f"Could not validate database {db_file}: {e}")
    
    def _check_ml_model_violations(self, model_file: Path):
        """Check ML model files for bulletproof violations."""
        try:
            with open(model_file, 'r', encoding='utf-8') as f:
                content = f.read()
                lines = content.split('\n')
            
            # Check for hardcoded label mappings
            hardcoded_patterns = [
                r'{\s*[\'"]background[\'"]\s*:\s*0\s*,.*[\'"]contradiction[\'"]\s*:\s*4\s*}',
                r'if\s+label_mappings\s+is\s+None\s*:.*=\s*{',
                r'default.*mapping.*=.*{.*background.*method.*support'
            ]
            
            for pattern in hardcoded_patterns:
                matches = re.finditer(pattern, content, re.IGNORECASE | re.DOTALL)
                for match in matches:
                    line_num = content[:match.start()].count('\n') + 1
                    
                    self.violations.append(Violation(
                        type="HARDCODED_ML_LABELS",
                        severity="critical",
                        file=str(model_file.relative_to(self.project_root)),
                        line=line_num,
                        column=0,
                        description="Hardcoded ML label mappings detected",
                        evidence=match.group()[:100] + "...",
                        suggestion="Remove hardcoded mappings, require expert-derived labels only"
                    ))
                    
        except Exception as e:
            self.logger.warning(f"Could not analyze ML model {model_file}: {e}")
    
    def _analyze_ast_tree(self, tree: ast.AST, file_path: Path, lines: List[str]):
        """Analyze AST tree for suspicious patterns."""
        
        class ViolationVisitor(ast.NodeVisitor):
            def __init__(self, analyzer, file_path, lines):
                self.analyzer = analyzer
                self.file_path = file_path
                self.lines = lines
                
            def visit_FunctionDef(self, node):
                # Check function names
                if any(forbidden in node.name.lower() 
                       for forbidden in self.analyzer.forbidden_patterns['simulation_keywords']):
                    self.analyzer.violations.append(Violation(
                        type="SUSPICIOUS_FUNCTION_NAME",
                        severity="high",
                        file=str(self.file_path.relative_to(self.analyzer.project_root)),
                        line=node.lineno,
                        column=node.col_offset,
                        description=f"Suspicious function name: '{node.name}'",
                        evidence=self.lines[node.lineno - 1].strip() if node.lineno <= len(self.lines) else "",
                        suggestion="Rename function to reflect real implementation"
                    ))
                
                self.generic_visit(node)
                
            def visit_Assign(self, node):
                # Check for suspicious assignments
                for target in node.targets:
                    if isinstance(target, ast.Name):
                        if any(suspicious in target.id.lower() 
                               for suspicious in self.analyzer.forbidden_patterns['suspicious_variables']):
                            self.analyzer.violations.append(Violation(
                                type="SUSPICIOUS_VARIABLE",
                                severity="medium",
                                file=str(self.file_path.relative_to(self.analyzer.project_root)),
                                line=node.lineno,
                                column=node.col_offset,
                                description=f"Suspicious variable assignment: '{target.id}'",
                                evidence=self.lines[node.lineno - 1].strip() if node.lineno <= len(self.lines) else "",
                                suggestion="Remove simulation-related variables"
                            ))
                
                self.generic_visit(node)
        
        visitor = ViolationVisitor(self, file_path, lines)
        visitor.visit(tree)
    
    def _should_skip_file(self, file_path: Path) -> bool:
        """Check if file should be skipped during analysis."""
        skip_patterns = [
            '__pycache__',
            '.git',
            '.pytest_cache',
            'node_modules',
            '.venv',
            'venv',
            'env',
            # Allow analysis of test files to ensure they don't contain production code
        ]
        
        return any(pattern in str(file_path) for pattern in skip_patterns)
    
    def _is_validation_script(self, file_path: Path) -> bool:
        """Check if this is a validation script that legitimately contains detection patterns."""
        validation_scripts = [
            'bulletproof_validator.py',
            'bulletproof_code_analyzer.py',
            'validate_training_data.py',
            'generate_compliance_report.py'
        ]
        return file_path.name in validation_scripts
    
    def _is_legitimate_pattern_usage(self, line: str) -> bool:
        """Check if line legitimately uses patterns for detection/rejection."""
        line_lower = line.lower()
        legitimate_contexts = [
            'forbidden_patterns',
            'suspicious_patterns',
            'simulation_indicators',
            'check for',
            'detect',
            'reject',
            'validation',
            'pattern',
            'bulletproof',
            'no simulation',
            '# ',  # Comments
            '"""',  # Docstrings
            "'''",  # Docstrings
            'raise',  # Error messages
            'logger.',  # Logging
            'print(',  # Print statements
            'description=',  # Descriptions
            'violation',
            'evidence',
            'suggestion'
        ]
        
        return any(context in line_lower for context in legitimate_contexts)
    
    def _violation_to_dict(self, violation: Violation) -> Dict[str, Any]:
        """Convert violation to dictionary for JSON serialization."""
        return {
            'type': violation.type,
            'severity': violation.severity,
            'file': violation.file,
            'line': violation.line,
            'column': violation.column,
            'description': violation.description,
            'evidence': violation.evidence,
            'suggestion': violation.suggestion
        }
    
    def _generate_summary(self) -> Dict[str, Any]:
        """Generate summary of violations."""
        summary = {
            'total_violations': len(self.violations),
            'by_severity': {},
            'by_type': {},
            'bulletproof_compliant': len(self.violations) == 0
        }
        
        # Count by severity
        for violation in self.violations:
            summary['by_severity'][violation.severity] = \
                summary['by_severity'].get(violation.severity, 0) + 1
            summary['by_type'][violation.type] = \
                summary['by_type'].get(violation.type, 0) + 1
                
        return summary


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Bulletproof Code Analyzer - Detect simulation/placeholder violations"
    )
    
    parser.add_argument('--mode', 
                       choices=['static', 'database', 'advanced', 'comprehensive'],
                       default='comprehensive',
                       help='Analysis mode')
    
    parser.add_argument('--fail-fast', action='store_true',
                       help='Exit immediately on first violation')
    
    parser.add_argument('--strict', action='store_true',
                       help='Use strictest validation rules')
    
    parser.add_argument('--ml-patterns', action='store_true',
                       help='Enable ML-specific pattern detection')
    
    parser.add_argument('--output', default='reports/violations.json',
                       help='Output file for violation report')
    
    args = parser.parse_args()
    
    # Create reports directory
    os.makedirs('reports', exist_ok=True)
    
    # Run analysis
    analyzer = BulletproofCodeAnalyzer()
    results = analyzer.analyze_codebase(mode=args.mode)
    
    # Save results
    with open(args.output, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Print summary
    print(f"\n{'='*80}")
    print("🛡️  BULLETPROOF CODE ANALYSIS RESULTS")
    print(f"{'='*80}")
    print(f"Analysis Mode: {results['analysis_mode']}")
    print(f"Total Violations: {results['summary']['total_violations']}")
    
    if results['summary']['total_violations'] > 0:
        print(f"\n❌ BULLETPROOF VALIDATION FAILED")
        print(f"Violations by severity: {results['summary']['by_severity']}")
        print(f"Violations by type: {results['summary']['by_type']}")
        
        # Show first few violations
        for i, violation in enumerate(results['violations'][:5]):
            print(f"\n🚨 VIOLATION {i+1}:")
            print(f"   Type: {violation['type']}")
            print(f"   File: {violation['file']}:{violation['line']}")
            print(f"   Description: {violation['description']}")
            print(f"   Evidence: {violation['evidence'][:100]}...")
            
        if len(results['violations']) > 5:
            print(f"\n... and {len(results['violations']) - 5} more violations")
            
        print(f"\n📋 Full report saved to: {args.output}")
        sys.exit(1)
        
    else:
        print(f"\n✅ BULLETPROOF VALIDATION PASSED")
        print("🎉 No violations detected - code maintains bulletproof standards!")
        sys.exit(0)


if __name__ == "__main__":
    main()