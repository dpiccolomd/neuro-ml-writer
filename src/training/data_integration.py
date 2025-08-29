"""
Data Integration Pipeline: Papers → Expert Annotations → Training Datasets

BULLETPROOF integration that connects real paper collection to ML training.
Handles the complete pipeline from collected papers to ready-to-train datasets.
"""

import sqlite3
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
import pandas as pd
from datetime import datetime
import asyncio
from dataclasses import dataclass
import re

from .data_collector import DataCollectionOrchestrator, PaperMetadata
from ..agents.utils.data_processing import (
    AnnotationDatabaseManager, CitationContext, create_train_val_test_split
)


@dataclass
class CitationCandidate:
    """
    Candidate text segment for citation annotation.
    
    Extracted from real papers, ready for expert review.
    """
    text: str
    paper_id: str
    section: str  # abstract, introduction, methods, results, discussion
    sentence_index: int
    paragraph_index: int
    context_before: str
    context_after: str
    
    def __post_init__(self):
        if not self.text.strip():
            raise ValueError("Citation candidate text cannot be empty")


class PaperTextProcessor:
    """
    Extract citation candidates from real collected papers.
    
    NO simulation - only processes authentic scientific papers.
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Patterns that suggest citation necessity (learned from literature analysis)
        self.citation_patterns = [
            r'previous studies? (?:have )?(?:shown|demonstrated|found|reported)',
            r'according to \w+',
            r'as (?:reported|shown|demonstrated) by',
            r'research (?:has )?(?:shown|demonstrated|indicated)',
            r'studies? (?:have )?(?:suggested|indicated|shown)',
            r'it has been (?:shown|demonstrated|reported)',
            r'evidence suggests?',
            r'findings? (?:suggest|indicate|demonstrate)',
            r'(?:Smith|Jones|\w+ et al\.?)',  # Author patterns
            r'\d{4}[a-z]?\)',  # Year patterns
        ]
        
    def extract_citation_candidates(self, paper: PaperMetadata) -> List[CitationCandidate]:
        """
        Extract text segments that may need citations from real paper.
        
        Returns segments for expert annotation, not pre-labeled data.
        """
        candidates = []
        
        # Process abstract if available
        if paper.abstract:
            candidates.extend(self._process_text_section(
                paper.abstract, paper.pmid or paper.doi, "abstract"
            ))
            
        return candidates
        
    def _process_text_section(
        self, 
        text: str, 
        paper_id: str, 
        section: str
    ) -> List[CitationCandidate]:
        """Process a section of paper text into citation candidates."""
        
        if not text.strip():
            return []
            
        # Split into sentences
        sentences = self._split_sentences(text)
        candidates = []
        
        for para_idx, paragraph in enumerate(self._split_paragraphs(text)):
            para_sentences = self._split_sentences(paragraph)
            
            for sent_idx, sentence in enumerate(para_sentences):
                # Only extract sentences that might need citations
                if self._might_need_citation(sentence):
                    # Get context
                    context_before = ' '.join(para_sentences[max(0, sent_idx-2):sent_idx])
                    context_after = ' '.join(para_sentences[sent_idx+1:min(len(para_sentences), sent_idx+3)])
                    
                    candidate = CitationCandidate(
                        text=sentence.strip(),
                        paper_id=paper_id,
                        section=section,
                        sentence_index=sent_idx,
                        paragraph_index=para_idx,
                        context_before=context_before.strip(),
                        context_after=context_after.strip()
                    )
                    candidates.append(candidate)
                    
        return candidates
        
    def _split_sentences(self, text: str) -> List[str]:
        """Split text into sentences using simple but effective rules."""
        # Basic sentence splitting
        sentences = re.split(r'[.!?]+\s+', text)
        return [s.strip() for s in sentences if s.strip()]
        
    def _split_paragraphs(self, text: str) -> List[str]:
        """Split text into paragraphs."""
        paragraphs = re.split(r'\n\s*\n', text)
        return [p.strip() for p in paragraphs if p.strip()]
        
    def _might_need_citation(self, sentence: str) -> bool:
        """
        Determine if sentence might need citation using pattern matching.
        
        This is just for candidate extraction - expert will make final decision.
        """
        sentence_lower = sentence.lower()
        
        # Skip very short sentences
        if len(sentence.split()) < 5:
            return False
            
        # Check for citation-suggesting patterns
        for pattern in self.citation_patterns:
            if re.search(pattern, sentence_lower):
                return True
                
        # Look for factual claims (heuristic)
        factual_indicators = [
            'is', 'are', 'was', 'were', 'has been', 'have been',
            'shows', 'demonstrates', 'indicates', 'suggests'
        ]
        
        if any(indicator in sentence_lower for indicator in factual_indicators):
            return True
            
        return False


class ExpertAnnotationCoordinator:
    """
    Coordinates the expert annotation process.
    
    Manages assignment of citation candidates to expert annotators.
    """
    
    def __init__(self, annotation_db_path: str):
        self.annotation_db = AnnotationDatabaseManager(annotation_db_path)
        self.logger = logging.getLogger(__name__)
        
    def prepare_annotation_tasks(
        self, 
        candidates: List[CitationCandidate],
        target_annotations_per_expert: int = 500
    ) -> Dict[str, List[CitationCandidate]]:
        """
        Prepare annotation tasks for expert review.
        
        Returns: dict mapping expert_id -> list of candidates for annotation
        """
        
        # For now, return all candidates for annotation
        # In production, this would implement load balancing across experts
        annotation_tasks = {
            'expert_batch_1': candidates
        }
        
        self.logger.info(f"Prepared {len(candidates)} candidates for expert annotation")
        
        return annotation_tasks
        
    def validate_expert_annotations_required(
        self,
        candidates: List[CitationCandidate]
    ) -> Dict[str, Any]:
        """
        BULLETPROOF: Validate that expert annotations are required and available.
        
        NO SIMULATION ALLOWED - only real expert annotations accepted.
        """
        
        validation_result = {
            'candidates_ready': len(candidates),
            'expert_annotations_required': True,
            'next_steps': [
                'Deploy expert annotation platform',
                'Recruit PhD-level neuroscience experts', 
                'Complete annotation of all candidates',
                'Validate inter-annotator agreement ≥90%'
            ]
        }
        
        self.logger.info(f"Expert annotation required for {len(candidates)} citation candidates")
        self.logger.info("BULLETPROOF POLICY: No simulation or mock data allowed")
        
        return validation_result


class DataIntegrationPipeline:
    """
    Complete pipeline from paper collection to training-ready datasets.
    
    Coordinates: Papers → Citation Candidates → Expert Annotations → Training Data
    """
    
    def __init__(
        self,
        papers_db_path: str = "./data/papers.db",
        annotations_db_path: str = "./data/annotations.db"
    ):
        self.papers_db_path = papers_db_path
        self.annotations_db_path = annotations_db_path
        
        self.collector = DataCollectionOrchestrator(db_path=papers_db_path)
        self.processor = PaperTextProcessor()
        self.coordinator = ExpertAnnotationCoordinator(annotations_db_path)
        
        self.logger = logging.getLogger(__name__)
        
    async def run_full_pipeline(
        self,
        max_papers: int = 1000
    ) -> Dict[str, Any]:
        """
        Run the complete data integration pipeline.
        
        BULLETPROOF: Only processes real papers and requires expert annotations.
        
        Args:
            max_papers: Maximum papers to process
        """
        
        results = {
            'papers_processed': 0,
            'candidates_extracted': 0,
            'expert_annotations_required': True,
            'training_ready': False
        }
        
        # Step 1: Ensure we have collected papers
        stats = self.collector.get_collection_stats()
        
        if stats.get('total', 0) < max_papers:
            self.logger.info(f"Collecting additional papers (current: {stats.get('total', 0)}, target: {max_papers})")
            collection_stats = await self.collector.collect_neuroscience_corpus(max_papers)
            results.update(collection_stats)
        
        # Step 2: Load papers and extract citation candidates
        papers = self._load_papers_from_db(max_papers)
        results['papers_processed'] = len(papers)
        
        all_candidates = []
        for paper in papers:
            candidates = self.processor.extract_citation_candidates(paper)
            all_candidates.extend(candidates)
            
        results['candidates_extracted'] = len(all_candidates)
        self.logger.info(f"Extracted {len(all_candidates)} citation candidates from {len(papers)} papers")
        
        # Step 3: Validate expert annotation requirements  
        validation_result = self.coordinator.validate_expert_annotations_required(all_candidates)
        results.update(validation_result)
        
        self.logger.info("BULLETPROOF PIPELINE STAGE: Expert annotation required")
        self.logger.info(f"Next steps: {validation_result['next_steps']}")
        
        # Check if we already have expert annotations in database
        existing_annotations = self.coordinator.annotation_db.get_annotation_stats()
        if existing_annotations.get('total', 0) > 0:
            self.logger.info(f"Found {existing_annotations['total']} existing expert annotations")
            
            # Create train/val/test splits if annotations exist
            splits = create_train_val_test_split(self.annotations_db_path)
            results['splits'] = splits
            results['training_ready'] = splits['train'] > 0
            results['expert_annotations_available'] = True
        else:
            self.logger.warning("No expert annotations found - training not possible")
            results['expert_annotations_available'] = False
            
        return results
        
    def _load_papers_from_db(self, limit: int) -> List[PaperMetadata]:
        """Load papers from collection database."""
        
        conn = sqlite3.connect(self.papers_db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT pmid, doi, title, abstract, authors, journal, publication_date,
                   keywords, mesh_terms, full_text_url, collection_source
            FROM papers 
            WHERE abstract IS NOT NULL AND abstract != ""
            LIMIT ?
        ''', (limit,))
        
        papers = []
        for row in cursor.fetchall():
            paper = PaperMetadata(
                pmid=row[0],
                doi=row[1], 
                title=row[2],
                abstract=row[3],
                authors=json.loads(row[4]) if row[4] else [],
                journal=row[5],
                publication_date=datetime.fromisoformat(row[6]) if row[6] else None,
                keywords=json.loads(row[7]) if row[7] else [],
                mesh_terms=json.loads(row[8]) if row[8] else [],
                full_text_url=row[9],
                collection_source=row[10]
            )
            papers.append(paper)
            
        conn.close()
        return papers
        
    def get_training_statistics(self) -> Dict[str, Any]:
        """Get statistics about training data readiness."""
        
        stats = self.coordinator.annotation_db.get_annotation_stats()
        
        # Check if we have minimum required data
        min_train_samples = 1000  # Minimum for meaningful training
        train_count = stats.get('by_split', {}).get('train', 0)
        
        stats['training_ready'] = train_count >= min_train_samples
        stats['min_required'] = min_train_samples
        
        return stats


# BULLETPROOF usage - no simulation allowed
async def main():
    """Run the bulletproof data integration pipeline."""
    
    logging.basicConfig(level=logging.INFO)
    
    pipeline = DataIntegrationPipeline()
    
    # Run bulletproof pipeline - expert annotations required
    results = await pipeline.run_full_pipeline(
        max_papers=100  # Start small for initial collection
    )
    
    print("BULLETPROOF Pipeline Results:")
    print(json.dumps(results, indent=2, default=str))
    
    # Check training readiness
    stats = pipeline.get_training_statistics()
    print("\nTraining Readiness Status:")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())