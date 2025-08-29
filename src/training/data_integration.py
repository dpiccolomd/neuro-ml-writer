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
        
    def simulate_expert_annotations(
        self, 
        candidates: List[CitationCandidate],
        annotator_id: str = "dev_annotator"
    ) -> List[CitationContext]:
        """
        TEMPORARY: Create mock expert annotations for development/testing.
        
        WARNING: This violates our bulletproof policy and should be replaced
        with real expert annotations before production training.
        """
        
        self.logger.warning(
            "USING SIMULATED ANNOTATIONS - Replace with real expert data before production!"
        )
        
        contexts = []
        for candidate in candidates:
            # Simple heuristics to simulate expert decisions (NOT for production)
            text_lower = candidate.text.lower()
            
            # Simulate citation necessity decision
            needs_citation = (
                any(word in text_lower for word in ['shown', 'found', 'reported', 'studies']) or
                'et al' in candidate.text or
                bool(re.search(r'\d{4}', candidate.text))
            )
            
            # Simulate citation type
            if 'method' in text_lower or 'procedure' in text_lower:
                citation_type = 'method'
            elif 'result' in text_lower or 'finding' in text_lower:
                citation_type = 'support'
            elif 'previous' in text_lower or 'prior' in text_lower:
                citation_type = 'background'
            else:
                citation_type = 'support'
                
            # Simulate placement
            citation_placement = 'end' if candidate.text.endswith('.') else 'middle'
            
            context = CitationContext(
                text=candidate.text,
                needs_citation=needs_citation,
                citation_type=citation_type,
                citation_placement=citation_placement,
                source_paper_id=candidate.paper_id,
                annotator_id=annotator_id,
                confidence_score=0.9  # High confidence for simulation
            )
            
            contexts.append(context)
            
        return contexts


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
        max_papers: int = 1000,
        use_simulation: bool = False
    ) -> Dict[str, Any]:
        """
        Run the complete data integration pipeline.
        
        Args:
            max_papers: Maximum papers to process
            use_simulation: If True, use simulated expert annotations (DEV ONLY)
        """
        
        results = {
            'papers_processed': 0,
            'candidates_extracted': 0,
            'annotations_created': 0,
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
        
        # Step 3: Create expert annotations
        if use_simulation:
            self.logger.warning("Using simulated annotations - NOT for production use!")
            contexts = self.coordinator.simulate_expert_annotations(all_candidates)
        else:
            # In production, this would coordinate real expert annotation
            self.logger.info(f"Ready for expert annotation: {len(all_candidates)} candidates")
            self.logger.info("Run expert annotation platform to label these candidates")
            return results
            
        # Step 4: Save annotations to database
        for context in contexts:
            success = self.coordinator.annotation_db.add_annotation(context)
            if success:
                results['annotations_created'] += 1
                
        # Step 5: Create train/val/test splits
        if results['annotations_created'] > 0:
            splits = create_train_val_test_split(self.annotations_db_path)
            results['splits'] = splits
            results['training_ready'] = splits['train'] > 0
            
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


# Example usage for development and testing
async def main():
    """Run the data integration pipeline."""
    
    logging.basicConfig(level=logging.INFO)
    
    pipeline = DataIntegrationPipeline()
    
    # Run with simulation for development (replace with real annotations)
    results = await pipeline.run_full_pipeline(
        max_papers=100,  # Start small for testing
        use_simulation=True  # REMOVE for production
    )
    
    print("Pipeline Results:")
    print(json.dumps(results, indent=2, default=str))
    
    # Check training readiness
    stats = pipeline.get_training_statistics()
    print("\nTraining Statistics:")
    print(json.dumps(stats, indent=2, default=str))


if __name__ == "__main__":
    asyncio.run(main())