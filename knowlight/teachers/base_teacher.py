from abc import ABC, abstractmethod


class BaseTeacher(ABC):
    """Abstract Base Class for Teacher Models in KnowLight framework."""

    @abstractmethod
    def extract_features(self, inputs):
        """Extract embeddings or logits from teacher model."""
        pass

    @abstractmethod
    def predict_ranking(self, user_profile, candidate_items):
        """Predict ranking scores for candidate items."""
        pass
