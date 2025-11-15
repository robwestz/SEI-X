"""
A/B testing framework for continuous improvement.
"""

from typing import List, Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass, field
from datetime import datetime, timedelta
import hashlib
import random
import numpy as np
from scipy import stats
import asyncio
from enum import Enum


class ExperimentStatus(Enum):
    """Experiment status."""
    DRAFT = "draft"
    RUNNING = "running"
    PAUSED = "paused"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class AllocationStrategy(Enum):
    """Traffic allocation strategy."""
    RANDOM = "random"
    DETERMINISTIC = "deterministic"
    ADAPTIVE = "adaptive"  # Multi-armed bandit


@dataclass
class Variant:
    """Experiment variant configuration."""
    name: str
    description: str
    config: Dict[str, Any]
    allocation: float = 0.5  # Traffic allocation percentage
    metrics: Dict[str, List[float]] = field(default_factory=dict)


@dataclass
class Experiment:
    """A/B test experiment."""
    experiment_id: str
    name: str
    description: str
    variants: List[Variant]
    metrics: List[str]
    start_time: Optional[datetime] = None
    end_time: Optional[datetime] = None
    status: ExperimentStatus = ExperimentStatus.DRAFT
    allocation_strategy: AllocationStrategy = AllocationStrategy.RANDOM
    minimum_sample_size: int = 1000
    confidence_level: float = 0.95
    metadata: Dict[str, Any] = field(default_factory=dict)


class ABTestingFramework:
    """Framework for conducting A/B tests on SIE-X."""

    def __init__(
            self,
            engine: 'SemanticIntelligenceEngine',
            storage_backend: Optional[Any] = None
    ):
        self.engine = engine
        self.storage = storage_backend
        self.experiments: Dict[str, Experiment] = {}
        self.active_experiments: List[str] = []
        self.allocation_cache: Dict[str, str] = {}

    def create_experiment(
            self,
            name: str,
            description: str,
            control_config: Dict[str, Any],
            treatment_configs: List[Tuple[str, Dict[str, Any]]],
            metrics: List[str],
            allocation_strategy: AllocationStrategy = AllocationStrategy.RANDOM,
            minimum_sample_size: int = 1000
    ) -> Experiment:
        """Create a new experiment."""
        experiment_id = f"exp_{int(datetime.now().timestamp())}"

        # Create variants
        variants = [
            Variant(
                name="control",
                description="Control group",
                config=control_config,
                allocation=0.5
            )
        ]

        # Add treatment variants
        treatment_allocation = 0.5 / len(treatment_configs)
        for treatment_name, treatment_config in treatment_configs:
            variants.append(Variant(
                name=treatment_name,
                description=f"Treatment: {treatment_name}",
                config=treatment_config,
                allocation=treatment_allocation
            ))

        experiment = Experiment(
            experiment_id=experiment_id,
            name=name,
            description=description,
            variants=variants,
            metrics=metrics,
            allocation_strategy=allocation_strategy,
            minimum_sample_size=minimum_sample_size
        )

        self.experiments[experiment_id] = experiment

        return experiment

    async def start_experiment(self, experiment_id: str):
        """Start an experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.status = ExperimentStatus.RUNNING
        experiment.start_time = datetime.now()
        self.active_experiments.append(experiment_id)

        logger.info(f"Started experiment {experiment_id}: {experiment.name}")

    async def stop_experiment(self, experiment_id: str):
        """Stop an experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            raise ValueError(f"Experiment {experiment_id} not found")

        experiment.status = ExperimentStatus.COMPLETED
        experiment.end_time = datetime.now()

        if experiment_id in self.active_experiments:
            self.active_experiments.remove(experiment_id)

        # Calculate results
        results = self.analyze_experiment(experiment)

        logger.info(f"Stopped experiment {experiment_id}: {results}")

        return results

    def get_variant(self, experiment_id: str, user_id: str) -> Optional[Variant]:
        """Get variant assignment for a user."""
        experiment = self.experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return None

        # Check cache
        cache_key = f"{experiment_id}:{user_id}"
        if cache_key in self.allocation_cache:
            variant_name = self.allocation_cache[cache_key]
            return next(v for v in experiment.variants if v.name == variant_name)

        # Allocate variant
        variant = self._allocate_variant(experiment, user_id)
        self.allocation_cache[cache_key] = variant.name

        return variant

    def _allocate_variant(self, experiment: Experiment, user_id: str) -> Variant:
        """Allocate user to variant based on strategy."""
        if experiment.allocation_strategy == AllocationStrategy.RANDOM:
            # Random allocation
            rand = random.random()
            cumulative = 0.0

            for variant in experiment.variants:
                cumulative += variant.allocation
                if rand < cumulative:
                    return variant

            return experiment.variants[-1]

        elif experiment.allocation_strategy == AllocationStrategy.DETERMINISTIC:
            # Hash-based deterministic allocation
            hash_value = int(hashlib.md5(
                f"{experiment.experiment_id}:{user_id}".encode()
            ).hexdigest(), 16)

            position = (hash_value % 100) / 100.0
            cumulative = 0.0

            for variant in experiment.variants:
                cumulative += variant.allocation
                if position < cumulative:
                    return variant

            return experiment.variants[-1]

        elif experiment.allocation_strategy == AllocationStrategy.ADAPTIVE:
            # Multi-armed bandit (Thompson sampling)
            return self._thompson_sampling(experiment)

        else:
            raise ValueError(f"Unknown allocation strategy: {experiment.allocation_strategy}")

    def _thompson_sampling(self, experiment: Experiment) -> Variant:
        """Thompson sampling for adaptive allocation."""
        # Calculate success rates for each variant
        variant_scores = []

        for variant in experiment.variants:
            successes = len([m for m in variant.metrics.get('success', []) if m > 0])
            failures = len(variant.metrics.get('success', [])) - successes

            # Beta distribution sampling
            if successes + failures > 0:
                score = np.random.beta(successes + 1, failures + 1)
            else:
                score = 0.5  # Prior

            variant_scores.append((variant, score))

        # Select variant with highest sampled score
        return max(variant_scores, key=lambda x: x[1])[0]

    async def track_metric(
            self,
            experiment_id: str,
            user_id: str,
            metric_name: str,
            value: float
    ):
        """Track metric for an experiment."""
        experiment = self.experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.RUNNING:
            return

        variant = self.get_variant(experiment_id, user_id)
        if not variant:
            return

        # Store metric
        if metric_name not in variant.metrics:
            variant.metrics[metric_name] = []

        variant.metrics[metric_name].append(value)

        # Check if experiment should end
        if await self._should_end_experiment(experiment):
            await self.stop_experiment(experiment_id)

    async def _should_end_experiment(self, experiment: Experiment) -> bool:
        """Check if experiment has reached stopping criteria."""
        # Check minimum sample size
        for variant in experiment.variants:
            sample_size = sum(len(metrics) for metrics in variant.metrics.values())
            if sample_size < experiment.minimum_sample_size:
                return False

        # Check statistical significance
        if len(experiment.variants) == 2:
            # Simple A/B test
            control = experiment.variants[0]
            treatment = experiment.variants[1]

            # Check primary metric
            primary_metric = experiment.metrics[0]

            if (primary_metric in control.metrics and
                    primary_metric in treatment.metrics):

                control_data = control.metrics[primary_metric]
                treatment_data = treatment.metrics[primary_metric]

                if len(control_data) > 30 and len(treatment_data) > 30:
                    # T-test
                    _, p_value = stats.ttest_ind(control_data, treatment_data)

                    # Early stopping if highly significant
                    if p_value < 0.001 or p_value > 0.999:
                        return True

        # Check time limit
        if experiment.start_time:
            duration = datetime.now() - experiment.start_time
            if duration > timedelta(days=30):  # Max 30 days
                return True

        return False

    def analyze_experiment(self, experiment: Experiment) -> Dict[str, Any]:
        """Analyze experiment results."""
        results = {
            'experiment_id': experiment.experiment_id,
            'name': experiment.name,
            'status': experiment.status.value,
            'duration': str(experiment.end_time - experiment.start_time) if experiment.end_time else None,
            'variants': {},
            'conclusions': []
        }

        # Analyze each variant
        for variant in experiment.variants:
            variant_stats = {
                'name': variant.name,
                'sample_size': sum(len(metrics) for metrics in variant.metrics.values()),
                'metrics': {}
            }

            for metric_name, values in variant.metrics.items():
                if values:
                    variant_stats['metrics'][metric_name] = {
                        'mean': np.mean(values),
                        'std': np.std(values),
                        'median': np.median(values),
                        'min': np.min(values),
                        'max': np.max(values),
                        'count': len(values)
                    }

            results['variants'][variant.name] = variant_stats

        # Statistical comparison
        if len(experiment.variants) == 2:
            control = experiment.variants[0]
            treatment = experiment.variants[1]

            for metric_name in experiment.metrics:
                if (metric_name in control.metrics and
                        metric_name in treatment.metrics):

                    control_data = control.metrics[metric_name]
                    treatment_data = treatment.metrics[metric_name]

                    if len(control_data) > 1 and len(treatment_data) > 1:
                        # Calculate statistics
                        t_stat, p_value = stats.ttest_ind(control_data, treatment_data)

                        # Effect size (Cohen's d)
                        pooled_std = np.sqrt(
                            ((len(control_data) - 1) * np.var(control_data) +
                             (len(treatment_data) - 1) * np.var(treatment_data)) /
                            (len(control_data) + len(treatment_data) - 2)
                        )

                        effect_size = (np.mean(treatment_data) - np.mean(control_data)) / pooled_std

                        # Confidence interval
                        ci = stats.t.interval(
                            experiment.confidence_level,
                            len(treatment_data) - 1,
                            loc=np.mean(treatment_data) - np.mean(control_data),
                            scale=stats.sem(treatment_data - np.mean(control_data))
                        )

                        comparison = {
                            'metric': metric_name,
                            't_statistic': float(t_stat),
                            'p_value': float(p_value),
                            'effect_size': float(effect_size),
                            'confidence_interval': [float(ci[0]), float(ci[1])],
                            'significant': p_value < (1 - experiment.confidence_level),
                            'improvement': float(
                                (np.mean(treatment_data) - np.mean(control_data)) / np.mean(control_data) * 100)
                        }

                        results['conclusions'].append(comparison)

        return results

    def get_winning_variant(self, experiment_id: str) -> Optional[str]:
        """Get the winning variant based on primary metric."""
        experiment = self.experiments.get(experiment_id)
        if not experiment or experiment.status != ExperimentStatus.COMPLETED:
            return None

        results = self.analyze_experiment(experiment)

        # Check primary metric
        if results['conclusions']:
            primary_conclusion = results['conclusions'][0]

            if primary_conclusion['significant']:
                if primary_conclusion['improvement'] > 0:
                    return experiment.variants[1].name  # Treatment wins
                else:
                    return experiment.variants[0].name  # Control wins

        return None  # No significant winner

    async def apply_winner(self, experiment_id: str):
        """Apply winning variant configuration to production."""
        winner = self.get_winning_variant(experiment_id)
        if not winner:
            logger.info(f"No winning variant for experiment {experiment_id}")
            return

        experiment = self.experiments[experiment_id]
        winning_variant = next(v for v in experiment.variants if v.name == winner)

        # Apply configuration
        for key, value in winning_variant.config.items():
            setattr(self.engine, key, value)

        logger.info(f"Applied winning variant '{winner}' from experiment {experiment_id}")


# Example experiments
class ExperimentLibrary:
    """Library of common experiments."""

    @staticmethod
    def embedding_model_experiment(framework: ABTestingFramework) -> Experiment:
        """Test different embedding models."""
        return framework.create_experiment(
            name="Embedding Model Comparison",
            description="Test performance of different sentence transformer models",
            control_config={
                'language_model': 'sentence-transformers/all-mpnet-base-v2'
            },
            treatment_configs=[
                ("minilm", {
                    'language_model': 'sentence-transformers/all-MiniLM-L12-v2'
                }),
                ("distilbert", {
                    'language_model': 'sentence-transformers/distilbert-base-nli-stsb-mean-tokens'
                })
            ],
            metrics=['latency', 'f1_score', 'user_satisfaction'],
            minimum_sample_size=5000
        )

    @staticmethod
    def chunking_strategy_experiment(framework: ABTestingFramework) -> Experiment:
        """Test different chunking strategies."""
        return framework.create_experiment(
            name="Chunking Strategy Test",
            description="Compare fixed vs adaptive chunking",
            control_config={
                'chunking_strategy': 'fixed',
                'max_chunk_size': 512
            },
            treatment_configs=[
                ("adaptive", {
                    'chunking_strategy': 'adaptive',
                    'max_chunk_size': 512,
                    'adaptive_size': True
                })
            ],
            metrics=['chunk_quality', 'processing_time', 'keyword_relevance'],
            allocation_strategy=AllocationStrategy.ADAPTIVE
        )