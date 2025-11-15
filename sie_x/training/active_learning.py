"""
Active learning system for continuous model improvement.
"""

import asyncio
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import precision_recall_fscore_support
import torch
from sentence_transformers import SentenceTransformer, InputExample, losses
from torch.utils.data import DataLoader


@dataclass
class FeedbackSample:
    """User feedback on extraction results."""
    text: str
    extracted_keywords: List[str]
    correct_keywords: List[str]
    user_rating: float
    timestamp: float
    context: Dict[str, Any]


class ActiveLearningPipeline:
    """Continuous learning from user feedback."""

    def __init__(
            self,
            base_model: SentenceTransformer,
            feedback_buffer_size: int = 1000,
            retrain_threshold: int = 100,
            uncertainty_threshold: float = 0.5
    ):
        self.base_model = base_model
        self.feedback_buffer: List[FeedbackSample] = []
        self.feedback_buffer_size = feedback_buffer_size
        self.retrain_threshold = retrain_threshold
        self.uncertainty_threshold = uncertainty_threshold
        self.model_version = 1
        self.performance_history = []

    async def add_feedback(self, feedback: FeedbackSample):
        """Add user feedback to buffer."""
        self.feedback_buffer.append(feedback)

        # Trigger retraining if buffer is full
        if len(self.feedback_buffer) >= self.retrain_threshold:
            await self.trigger_retraining()

    async def trigger_retraining(self):
        """Initiate model retraining with accumulated feedback."""
        logger.info(f"Starting active learning cycle with {len(self.feedback_buffer)} samples")

        # Prepare training data
        training_samples = self._prepare_training_data()

        # Calculate current performance
        current_metrics = self._evaluate_current_model(training_samples)

        # Fine-tune model
        improved_model = await self._fine_tune_model(training_samples)

        # Evaluate improvement
        new_metrics = self._evaluate_model(improved_model, training_samples)

        # Decide whether to deploy
        if self._should_deploy(current_metrics, new_metrics):
            await self._deploy_new_model(improved_model)
            logger.info(f"Deployed improved model v{self.model_version}")
        else:
            logger.warning("New model did not show improvement, keeping current")

        # Clear buffer
        self.feedback_buffer = []

    def _prepare_training_data(self) -> List[InputExample]:
        """Convert feedback to training examples."""
        examples = []

        for feedback in self.feedback_buffer:
            # Positive examples (correct keywords)
            for keyword in feedback.correct_keywords:
                examples.append(InputExample(
                    texts=[feedback.text, keyword],
                    label=1.0
                ))

            # Negative examples (incorrect extractions)
            incorrect = set(feedback.extracted_keywords) - set(feedback.correct_keywords)
            for keyword in incorrect:
                examples.append(InputExample(
                    texts=[feedback.text, keyword],
                    label=0.0
                ))

            # Hard negatives (similar but wrong)
            hard_negatives = self._generate_hard_negatives(
                feedback.text,
                feedback.correct_keywords
            )
            for neg in hard_negatives:
                examples.append(InputExample(
                    texts=[feedback.text, neg],
                    label=0.0
                ))

        return examples

    def _generate_hard_negatives(self, text: str, correct_keywords: List[str]) -> List[str]:
        """Generate challenging negative examples."""
        # This would use the current model to find similar but incorrect terms
        # Simplified implementation
        return []

    async def _fine_tune_model(self, training_samples: List[InputExample]) -> SentenceTransformer:
        """Fine-tune the model with new samples."""
        # Create a copy of the model
        new_model = SentenceTransformer(self.base_model._modules)

        # Prepare DataLoader
        train_dataloader = DataLoader(
            training_samples,
            shuffle=True,
            batch_size=16
        )

        # Define loss
        train_loss = losses.ContrastiveLoss(new_model)

        # Fine-tune
        new_model.fit(
            train_objectives=[(train_dataloader, train_loss)],
            epochs=3,
            warmup_steps=100,
            evaluation_steps=500,
            output_path=f'models/active_learning_v{self.model_version + 1}'
        )

        return new_model

    def _should_deploy(self, current_metrics: Dict, new_metrics: Dict) -> bool:
        """Decide if new model should be deployed."""
        # Compare key metrics
        improvement = (
                new_metrics['f1_score'] > current_metrics['f1_score'] * 1.02 and
                new_metrics['precision'] >= current_metrics['precision'] * 0.98
        )

        # Store performance history
        self.performance_history.append({
            'version': self.model_version,
            'metrics': current_metrics,
            'timestamp': asyncio.get_event_loop().time()
        })

        return improvement

    async def _deploy_new_model(self, model: SentenceTransformer):
        """Deploy the improved model."""
        self.base_model = model
        self.model_version += 1

        # Save model
        model.save(f'models/production_v{self.model_version}')

        # Update model registry
        await self._update_model_registry()

    def get_uncertainty_samples(self, candidates: List[str], embeddings: np.ndarray) -> List[Tuple[str, float]]:
        """Identify samples with high uncertainty for human review."""
        uncertainties = []

        # Calculate embedding variance as uncertainty proxy
        embedding_std = np.std(embeddings, axis=1)

        for i, candidate in enumerate(candidates):
            uncertainty = embedding_std[i].mean()
            if uncertainty > self.uncertainty_threshold:
                uncertainties.append((candidate, float(uncertainty)))

        return sorted(uncertainties, key=lambda x: x[1], reverse=True)