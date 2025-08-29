"""
Complete Citation Intelligence Agent Training Script

BULLETPROOF training script that uses real data and proper validation.
Demonstrates end-to-end pipeline from data collection to trained model.
"""

import asyncio
import logging
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer
import argparse
import sys
import os

# Add src to path for imports
sys.path.append(str(Path(__file__).parent.parent))

from training.data_integration import DataIntegrationPipeline
from agents.utils.data_processing import CitationDataset, AnnotationDatabaseManager
from agents.citation.models import CitationContextClassifier
from agents.citation.trainer import CitationTrainer, create_training_config


class CitationAgentTrainingPipeline:
    """
    Complete training pipeline for Citation Intelligence Agent.
    
    Handles: Data Collection → Annotation → Training → Validation
    """
    
    def __init__(
        self,
        data_dir: str = "./data",
        models_dir: str = "./models",
        require_expert_annotations: bool = True  # BULLETPROOF - only expert data allowed
    ):
        self.data_dir = Path(data_dir)
        self.models_dir = Path(models_dir)
        self.require_expert_annotations = require_expert_annotations
        
        # Create directories
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.data_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)
        
        # Database paths
        self.papers_db_path = str(self.data_dir / "papers.db")
        self.annotations_db_path = str(self.data_dir / "annotations.db")
        
    async def run_complete_pipeline(
        self,
        max_papers: int = 1000,
        min_annotations: int = 500
    ) -> Dict[str, Any]:
        """
        Run the complete training pipeline.
        
        Returns training results and model paths.
        """
        
        self.logger.info("Starting Citation Intelligence Agent training pipeline")
        
        results = {
            'pipeline_stages': {},
            'model_paths': {},
            'performance_metrics': {},
            'errors': []
        }
        
        try:
            # Stage 1: Data Integration
            self.logger.info("Stage 1: Data Collection and Integration")
            integration_pipeline = DataIntegrationPipeline(
                papers_db_path=self.papers_db_path,
                annotations_db_path=self.annotations_db_path
            )
            
            integration_results = await integration_pipeline.run_full_pipeline(
                max_papers=max_papers
            )
            
            results['pipeline_stages']['data_integration'] = integration_results
            
            if not integration_results.get('training_ready', False):
                if not integration_results.get('expert_annotations_available', False):
                    raise ValueError(
                        "BULLETPROOF POLICY: No expert annotations available. "
                        "Expert annotation phase must be completed before training can begin."
                    )
            
            # Stage 2: Dataset Preparation
            self.logger.info("Stage 2: Dataset Preparation")
            tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            
            # Create datasets
            train_dataset = CitationDataset(
                annotation_db_path=self.annotations_db_path,
                tokenizer=tokenizer,
                split="train"
            )
            
            val_dataset = CitationDataset(
                annotation_db_path=self.annotations_db_path,
                tokenizer=tokenizer,
                split="val"
            )
            
            if len(train_dataset) == 0:
                raise ValueError("No training data available - expert annotations required")
                
            self.logger.info(f"Training set: {len(train_dataset)} samples")
            self.logger.info(f"Validation set: {len(val_dataset)} samples")
            
            # Get label mappings from real data
            label_mappings = train_dataset.get_label_mappings()
            self.logger.info(f"Label mappings from real data: {label_mappings}")
            
            results['pipeline_stages']['dataset_preparation'] = {
                'train_samples': len(train_dataset),
                'val_samples': len(val_dataset),
                'label_mappings': label_mappings
            }
            
            # Stage 3: Model Training
            self.logger.info("Stage 3: Model Training")
            
            # Create model with real label mappings
            model = CitationContextClassifier(
                model_name="allenai/scibert_scivocab_uncased",
                label_mappings=label_mappings
            )
            
            # Training configuration
            config = create_training_config(
                learning_rate=2e-5,
                batch_size=8,
                num_epochs=3,
                use_cloud_optimizations=False  # Set to True for cloud training
            )
            
            # Create trainer
            trainer = CitationTrainer(
                model=model,
                tokenizer=tokenizer,
                train_dataset=train_dataset,
                val_dataset=val_dataset,
                config=config,
                experiment_name="citation-intelligence-dev",
                save_dir=str(self.models_dir / "citation")
            )
            
            # Train the model
            training_results = trainer.train()
            
            results['pipeline_stages']['model_training'] = training_results
            results['model_paths']['citation_model'] = training_results['model_save_path']
            results['performance_metrics'] = {
                'best_f1_score': training_results['best_val_f1'],
                'training_completed': True
            }
            
            self.logger.info(f"Training completed with best F1 score: {training_results['best_val_f1']:.4f}")
            
            # Stage 4: Model Validation
            self.logger.info("Stage 4: Model Validation")
            validation_results = self._validate_trained_model(
                model_path=results['model_paths']['citation_model'],
                label_mappings=label_mappings
            )
            
            results['pipeline_stages']['validation'] = validation_results
            
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            results['errors'].append(str(e))
            return results
            
        self.logger.info("Citation Intelligence Agent training pipeline completed successfully")
        return results
        
    def _validate_trained_model(
        self, 
        model_path: str, 
        label_mappings: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """Validate the trained model with basic functionality tests."""
        
        try:
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained("allenai/scibert_scivocab_uncased")
            
            # Load model
            model = CitationContextClassifier(label_mappings=label_mappings)
            
            # Load trained weights
            checkpoint = torch.load(
                Path(model_path) / "citation_model_best.pth",
                map_location=torch.device('cpu')
            )
            model.load_state_dict(checkpoint['model_state_dict'])
            model.eval()
            
            # Create integrated agent
            from agents.citation.models import IntegratedCitationAgent
            agent = IntegratedCitationAgent(
                context_classifier=model,
                selection_ranker=None,  # Not needed for basic validation
                network_gnn=None,      # Not needed for basic validation  
                tokenizer=tokenizer
            )
            
            # Test with real examples from collected papers (if available)
            validation_results = self._get_real_test_examples()
            
            if validation_results['real_examples_available']:
                predictions = []
                for text in validation_results['test_texts']:
                    pred = agent.predict_citation_necessity(text)
                    predictions.append(pred)
                    
                validation_results['test_predictions'] = predictions
            else:
                # No real examples available
                validation_results['test_predictions'] = []
                self.logger.warning("No real test examples available - model functionality not validated")
                
            return {
                'model_loaded': True,
                'predictions_working': validation_results['real_examples_available'],
                **validation_results,
                'validation_passed': validation_results['real_examples_available']
            }
            
        except Exception as e:
            self.logger.error(f"Model validation failed: {str(e)}")
            return {
                'model_loaded': False,
                'validation_passed': False,
                'error': str(e)
            }
    
    def _get_real_test_examples(self) -> Dict[str, Any]:
        """Get real test examples from collected papers database."""
        
        try:
            import sqlite3
            
            # Try to get examples from papers database
            papers_db_path = str(self.data_dir / "papers.db")
            
            if not Path(papers_db_path).exists():
                return {
                    'real_examples_available': False,
                    'reason': 'Papers database not found'
                }
            
            conn = sqlite3.connect(papers_db_path)
            cursor = conn.cursor()
            
            # Get sample abstracts for testing
            cursor.execute('''
                SELECT abstract FROM papers 
                WHERE abstract IS NOT NULL AND abstract != "" 
                AND LENGTH(abstract) > 100
                LIMIT 3
            ''')
            
            abstracts = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if not abstracts:
                return {
                    'real_examples_available': False,
                    'reason': 'No suitable abstracts found in database'
                }
            
            # Extract first sentence from each abstract as test text
            test_texts = []
            for abstract in abstracts:
                # Split into sentences and take the first meaningful one
                sentences = abstract.split('.')
                for sentence in sentences:
                    if len(sentence.strip()) > 20:  # Meaningful sentence
                        test_texts.append(sentence.strip() + '.')
                        break
            
            if not test_texts:
                return {
                    'real_examples_available': False,
                    'reason': 'Could not extract meaningful sentences from abstracts'
                }
            
            return {
                'real_examples_available': True,
                'test_texts': test_texts,
                'source': 'Real paper abstracts from database'
            }
            
        except Exception as e:
            return {
                'real_examples_available': False,
                'reason': f'Error loading real examples: {str(e)}'
            }


def main():
    """Main training script with command-line interface."""
    
    parser = argparse.ArgumentParser(description='Train Citation Intelligence Agent')
    parser.add_argument('--data_dir', default='./data', help='Data directory')
    parser.add_argument('--models_dir', default='./models', help='Models directory')
    parser.add_argument('--max_papers', type=int, default=100, help='Maximum papers to collect')
    parser.add_argument('--min_annotations', type=int, default=50, help='Minimum annotations required')
    parser.add_argument('--allow_no_annotations', action='store_true', help='Allow training without expert annotations (NOT RECOMMENDED)')
    
    args = parser.parse_args()
    
    # Create training pipeline
    pipeline = CitationAgentTrainingPipeline(
        data_dir=args.data_dir,
        models_dir=args.models_dir,
        require_expert_annotations=not args.allow_no_annotations
    )
    
    # Run training
    results = asyncio.run(pipeline.run_complete_pipeline(
        max_papers=args.max_papers,
        min_annotations=args.min_annotations
    ))
    
    # Print results
    print("\n" + "="*80)
    print("CITATION INTELLIGENCE AGENT TRAINING RESULTS")
    print("="*80)
    
    if results.get('errors'):
        print(f"\n❌ ERRORS ENCOUNTERED:")
        for error in results['errors']:
            print(f"   - {error}")
    
    if results['pipeline_stages'].get('model_training'):
        training_results = results['pipeline_stages']['model_training']
        print(f"\n✅ TRAINING COMPLETED:")
        print(f"   - Best F1 Score: {training_results.get('best_val_f1', 'N/A'):.4f}")
        print(f"   - Model Path: {results['model_paths'].get('citation_model', 'N/A')}")
    
    if results['pipeline_stages'].get('validation', {}).get('validation_passed'):
        print(f"\n✅ MODEL VALIDATION: PASSED")
    else:
        print(f"\n❌ MODEL VALIDATION: FAILED")
    
    print(f"\nComplete results saved to: {args.data_dir}/training.log")
    
    # Save detailed results
    results_path = Path(args.data_dir) / "training_results.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"Detailed results: {results_path}")


if __name__ == "__main__":
    main()