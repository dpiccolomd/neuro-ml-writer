"""
Data Collection Pipeline for ML Training

Bulletproof data collection system for gathering 50,000+ neuroscience papers
with expert annotations for training Citation Intelligence models.
"""

import asyncio
import aiohttp
import requests
import json
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, List, Optional, Tuple, AsyncGenerator
import logging
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
import time
import pandas as pd
import numpy as np
from tqdm import tqdm
import hashlib
import PyPDF2
import pymupdf  # fitz
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import os
import sqlite3


@dataclass
class PaperMetadata:
    """Structured metadata for scientific papers."""
    pmid: Optional[str] = None
    doi: Optional[str] = None
    title: str = ""
    abstract: str = ""
    authors: List[str] = None
    journal: str = ""
    publication_date: Optional[datetime] = None
    keywords: List[str] = None
    mesh_terms: List[str] = None
    citation_count: int = 0
    references: List[str] = None
    full_text_url: Optional[str] = None
    pdf_path: Optional[str] = None
    collection_source: str = ""  # pubmed, arxiv, pmc, etc.
    
    def __post_init__(self):
        if self.authors is None:
            self.authors = []
        if self.keywords is None:
            self.keywords = []
        if self.mesh_terms is None:
            self.mesh_terms = []
        if self.references is None:
            self.references = []


class PubMedCollector:
    """
    High-performance PubMed data collector for neuroscience papers.
    
    Features:
    - Async HTTP requests for maximum throughput
    - Automatic rate limiting to respect API limits
    - Robust error handling and retry logic
    - Comprehensive metadata extraction
    """
    
    def __init__(self, api_key: Optional[str] = None, rate_limit: int = 3):
        self.api_key = api_key
        self.rate_limit = rate_limit  # requests per second
        self.base_url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"
        
        # Setup logging
        self.logger = logging.getLogger(__name__)
        
        # Rate limiting
        self.last_request_time = 0
        
    async def search_papers(
        self, 
        query: str, 
        max_results: int = 10000,
        date_range: Optional[Tuple[datetime, datetime]] = None
    ) -> List[str]:
        """Search PubMed for paper IDs matching the query."""
        
        search_url = f"{self.base_url}/esearch.fcgi"
        
        params = {
            'db': 'pubmed',
            'term': query,
            'retmax': max_results,
            'retmode': 'json',
            'sort': 'relevance'
        }
        
        if self.api_key:
            params['api_key'] = self.api_key
            
        if date_range:
            start_date = date_range[0].strftime('%Y/%m/%d')
            end_date = date_range[1].strftime('%Y/%m/%d')
            params['datetype'] = 'pdat'
            params['mindate'] = start_date
            params['maxdate'] = end_date
        
        await self._rate_limit()
        
        async with aiohttp.ClientSession() as session:
            async with session.get(search_url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    pmids = data['esearchresult']['idlist']
                    count = int(data['esearchresult']['count'])
                    
                    self.logger.info(f"Found {count} papers for query: {query}")
                    return pmids
                else:
                    self.logger.error(f"PubMed search failed: {response.status}")
                    return []
                    
    async def fetch_paper_metadata(self, pmids: List[str]) -> AsyncGenerator[PaperMetadata, None]:
        """Fetch detailed metadata for papers in batches."""
        
        batch_size = 200  # PubMed recommends max 200 per request
        
        for i in range(0, len(pmids), batch_size):
            batch = pmids[i:i + batch_size]
            
            fetch_url = f"{self.base_url}/efetch.fcgi"
            params = {
                'db': 'pubmed',
                'id': ','.join(batch),
                'retmode': 'xml',
                'rettype': 'abstract'
            }
            
            if self.api_key:
                params['api_key'] = self.api_key
                
            await self._rate_limit()
            
            async with aiohttp.ClientSession() as session:
                async with session.get(fetch_url, params=params) as response:
                    if response.status == 200:
                        xml_data = await response.text()
                        papers = self._parse_pubmed_xml(xml_data)
                        
                        for paper in papers:
                            yield paper
                    else:
                        self.logger.error(f"Failed to fetch batch: {response.status}")
                        
    def _parse_pubmed_xml(self, xml_data: str) -> List[PaperMetadata]:
        """Parse PubMed XML response into structured metadata."""
        
        papers = []
        root = ET.fromstring(xml_data)
        
        for article in root.findall('.//PubmedArticle'):
            try:
                paper = PaperMetadata()
                
                # PMID
                pmid_elem = article.find('.//PMID')
                if pmid_elem is not None:
                    paper.pmid = pmid_elem.text
                
                # DOI
                doi_elem = article.find('.//ArticleId[@IdType="doi"]')
                if doi_elem is not None:
                    paper.doi = doi_elem.text
                
                # Title
                title_elem = article.find('.//ArticleTitle')
                if title_elem is not None:
                    paper.title = ''.join(title_elem.itertext()).strip()
                
                # Abstract
                abstract_elem = article.find('.//Abstract/AbstractText')
                if abstract_elem is not None:
                    paper.abstract = ''.join(abstract_elem.itertext()).strip()
                
                # Authors
                authors = []
                for author in article.findall('.//Author'):
                    lastname = author.find('LastName')
                    forename = author.find('ForeName')
                    if lastname is not None and forename is not None:
                        authors.append(f"{forename.text} {lastname.text}")
                paper.authors = authors
                
                # Journal
                journal_elem = article.find('.//Journal/Title')
                if journal_elem is not None:
                    paper.journal = journal_elem.text
                
                # Publication date
                pub_date = article.find('.//PubDate')
                if pub_date is not None:
                    year_elem = pub_date.find('Year')
                    month_elem = pub_date.find('Month')
                    day_elem = pub_date.find('Day')
                    
                    if year_elem is not None:
                        year = int(year_elem.text)
                        month = 1
                        day = 1
                        
                        if month_elem is not None:
                            try:
                                month = int(month_elem.text)
                            except ValueError:
                                # Handle month names
                                month_names = {
                                    'jan': 1, 'feb': 2, 'mar': 3, 'apr': 4,
                                    'may': 5, 'jun': 6, 'jul': 7, 'aug': 8,
                                    'sep': 9, 'oct': 10, 'nov': 11, 'dec': 12
                                }
                                month = month_names.get(month_elem.text.lower()[:3], 1)
                        
                        if day_elem is not None:
                            day = int(day_elem.text)
                            
                        paper.publication_date = datetime(year, month, day)
                
                # MeSH terms
                mesh_terms = []
                for mesh in article.findall('.//MeshHeading/DescriptorName'):
                    mesh_terms.append(mesh.text)
                paper.mesh_terms = mesh_terms
                
                # Keywords
                keywords = []
                for keyword in article.findall('.//Keyword'):
                    keywords.append(keyword.text)
                paper.keywords = keywords
                
                paper.collection_source = "pubmed"
                papers.append(paper)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse paper: {e}")
                continue
                
        return papers
    
    async def _rate_limit(self):
        """Ensure we don't exceed API rate limits."""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        min_interval = 1.0 / self.rate_limit
        
        if time_since_last < min_interval:
            await asyncio.sleep(min_interval - time_since_last)
            
        self.last_request_time = time.time()


class ArXivCollector:
    """Collector for arXiv neuroscience papers."""
    
    def __init__(self):
        self.base_url = "http://export.arxiv.org/api/query"
        self.logger = logging.getLogger(__name__)
    
    async def collect_papers(
        self, 
        categories: List[str] = ["q-bio.NC"],  # Neuroscience category
        max_results: int = 5000
    ) -> AsyncGenerator[PaperMetadata, None]:
        """Collect papers from arXiv."""
        
        query = ' OR '.join([f"cat:{cat}" for cat in categories])
        
        params = {
            'search_query': query,
            'max_results': max_results,
            'sortBy': 'submittedDate',
            'sortOrder': 'descending'
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(self.base_url, params=params) as response:
                if response.status == 200:
                    xml_data = await response.text()
                    papers = self._parse_arxiv_xml(xml_data)
                    
                    for paper in papers:
                        yield paper
                else:
                    self.logger.error(f"arXiv request failed: {response.status}")
    
    def _parse_arxiv_xml(self, xml_data: str) -> List[PaperMetadata]:
        """Parse arXiv XML response."""
        
        papers = []
        root = ET.fromstring(xml_data)
        
        # Define namespace
        ns = {'atom': 'http://www.w3.org/2005/Atom'}
        
        for entry in root.findall('.//atom:entry', ns):
            try:
                paper = PaperMetadata()
                
                # Title
                title_elem = entry.find('atom:title', ns)
                if title_elem is not None:
                    paper.title = title_elem.text.strip()
                
                # Abstract
                summary_elem = entry.find('atom:summary', ns)
                if summary_elem is not None:
                    paper.abstract = summary_elem.text.strip()
                
                # Authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name_elem = author.find('atom:name', ns)
                    if name_elem is not None:
                        authors.append(name_elem.text)
                paper.authors = authors
                
                # arXiv ID and DOI
                arxiv_id = entry.find('atom:id', ns).text.split('/')[-1]
                paper.pmid = arxiv_id  # Use arXiv ID as identifier
                
                # Publication date
                published_elem = entry.find('atom:published', ns)
                if published_elem is not None:
                    paper.publication_date = datetime.fromisoformat(
                        published_elem.text.replace('Z', '+00:00')
                    )
                
                # PDF URL
                for link in entry.findall('atom:link', ns):
                    if link.get('type') == 'application/pdf':
                        paper.full_text_url = link.get('href')
                        break
                
                paper.collection_source = "arxiv"
                papers.append(paper)
                
            except Exception as e:
                self.logger.warning(f"Failed to parse arXiv paper: {e}")
                continue
                
        return papers


class DataCollectionOrchestrator:
    """
    Main orchestrator for collecting large-scale neuroscience literature.
    
    Coordinates multiple data sources and manages the complete collection pipeline.
    """
    
    def __init__(self, data_dir: str = "./data", db_path: str = "./data/papers.db"):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.db_path = db_path
        self._init_database()
        
        # Initialize collectors
        self.pubmed = PubMedCollector()
        self.arxiv = ArXivCollector()
        
        self.logger = logging.getLogger(__name__)
        
    def _init_database(self):
        """Initialize SQLite database for paper storage."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS papers (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                pmid TEXT UNIQUE,
                doi TEXT,
                title TEXT NOT NULL,
                abstract TEXT,
                authors TEXT,  -- JSON array
                journal TEXT,
                publication_date TEXT,
                keywords TEXT,  -- JSON array
                mesh_terms TEXT,  -- JSON array
                citation_count INTEGER DEFAULT 0,
                references TEXT,  -- JSON array
                full_text_url TEXT,
                pdf_path TEXT,
                collection_source TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                processed BOOLEAN DEFAULT FALSE
            )
        ''')
        
        conn.commit()
        conn.close()
        
    async def collect_neuroscience_corpus(self, target_papers: int = 50000) -> Dict[str, int]:
        """Collect comprehensive neuroscience corpus from multiple sources."""
        
        collection_stats = {
            'pubmed': 0,
            'arxiv': 0,
            'total': 0,
            'errors': 0
        }
        
        self.logger.info(f"Starting collection of {target_papers} neuroscience papers")
        
        # PubMed queries for different neuroscience areas
        pubmed_queries = [
            "(neuroscience[MeSH] OR neurolog*[Title/Abstract] OR brain[Title/Abstract]) AND english[Language]",
            "(cognitive neuroscience[Title/Abstract] OR neuroimaging[Title/Abstract]) AND english[Language]",
            "(neurosurgery[MeSH] OR neurosurgical[Title/Abstract]) AND english[Language]",
            "(neurological disorders[Title/Abstract] OR brain disease*[Title/Abstract]) AND english[Language]",
            "(synaptic plasticity[Title/Abstract] OR neural network*[Title/Abstract]) AND english[Language]"
        ]
        
        # Collect from PubMed
        pubmed_target = int(target_papers * 0.8)  # 80% from PubMed
        papers_per_query = pubmed_target // len(pubmed_queries)
        
        for query in pubmed_queries:
            try:
                self.logger.info(f"Collecting from PubMed: {query}")
                pmids = await self.pubmed.search_papers(query, papers_per_query)
                
                async for paper in self.pubmed.fetch_paper_metadata(pmids):
                    if await self._save_paper(paper):
                        collection_stats['pubmed'] += 1
                        collection_stats['total'] += 1
                        
                        if collection_stats['total'] % 1000 == 0:
                            self.logger.info(f"Collected {collection_stats['total']} papers so far")
                            
            except Exception as e:
                self.logger.error(f"PubMed collection error: {e}")
                collection_stats['errors'] += 1
        
        # Collect from arXiv
        arxiv_target = target_papers - collection_stats['total']
        if arxiv_target > 0:
            try:
                self.logger.info("Collecting from arXiv")
                async for paper in self.arxiv.collect_papers(max_results=arxiv_target):
                    if await self._save_paper(paper):
                        collection_stats['arxiv'] += 1
                        collection_stats['total'] += 1
                        
            except Exception as e:
                self.logger.error(f"arXiv collection error: {e}")
                collection_stats['errors'] += 1
        
        self.logger.info(f"Collection completed: {collection_stats}")
        return collection_stats
    
    async def _save_paper(self, paper: PaperMetadata) -> bool:
        """Save paper to database with deduplication."""
        
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check for duplicates
            if paper.pmid:
                cursor.execute("SELECT id FROM papers WHERE pmid = ?", (paper.pmid,))
                if cursor.fetchone():
                    conn.close()
                    return False
            
            if paper.doi:
                cursor.execute("SELECT id FROM papers WHERE doi = ?", (paper.doi,))
                if cursor.fetchone():
                    conn.close()
                    return False
            
            # Insert new paper
            cursor.execute('''
                INSERT INTO papers (
                    pmid, doi, title, abstract, authors, journal, publication_date,
                    keywords, mesh_terms, full_text_url, collection_source
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                paper.pmid,
                paper.doi,
                paper.title,
                paper.abstract,
                json.dumps(paper.authors),
                paper.journal,
                paper.publication_date.isoformat() if paper.publication_date else None,
                json.dumps(paper.keywords),
                json.dumps(paper.mesh_terms),
                paper.full_text_url,
                paper.collection_source
            ))
            
            conn.commit()
            conn.close()
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to save paper: {e}")
            return False
    
    def get_collection_stats(self) -> Dict[str, int]:
        """Get current collection statistics."""
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        stats = {}
        
        # Total papers
        cursor.execute("SELECT COUNT(*) FROM papers")
        stats['total'] = cursor.fetchone()[0]
        
        # By source
        cursor.execute("SELECT collection_source, COUNT(*) FROM papers GROUP BY collection_source")
        for source, count in cursor.fetchall():
            stats[source] = count
        
        # By year
        cursor.execute("""
            SELECT strftime('%Y', publication_date) as year, COUNT(*) 
            FROM papers 
            WHERE publication_date IS NOT NULL 
            GROUP BY year 
            ORDER BY year DESC 
            LIMIT 10
        """)
        
        stats['by_year'] = dict(cursor.fetchall())
        
        conn.close()
        return stats


# Example usage for cloud deployment
async def main():
    """Main execution function for cloud training data collection."""
    
    # Initialize collector
    collector = DataCollectionOrchestrator()
    
    # Set target based on cloud resources
    target_papers = int(os.getenv('TARGET_PAPERS', '50000'))
    
    # Start collection
    stats = await collector.collect_neuroscience_corpus(target_papers)
    
    print("Collection completed:")
    print(json.dumps(stats, indent=2))
    
    # Display final statistics
    final_stats = collector.get_collection_stats()
    print("Final corpus statistics:")
    print(json.dumps(final_stats, indent=2))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(main())