from __future__ import annotations

from collections.abc import Iterable
from pathlib import Path
from typing import Any

from catboost import CatBoostClassifier


class Model:
    def __init__(self, classifier: CatBoostClassifier) -> None:
        self._classifier = classifier

    @classmethod
    def load(cls, path: Path) -> Model:
        classifier = CatBoostClassifier()
        classifier.load_model(str(path))
        return cls(classifier)

    def feature_names(self) -> list[str]:
        return self._classifier.feature_names_

    def validate_feature_names(self, feature_names: Iterable[str]) -> None:
        """Raise ValueError if the given feature names don't match this model's inputs exactly."""
        provided = set(feature_names)
        expected = set(self.feature_names())
        missing = expected - provided
        extra = provided - expected
        if missing or extra:
            raise ValueError(
                f"Feature name mismatch: missing={sorted(missing)}, extra={sorted(extra)}"
            )

    def predict(self, X: Any) -> Any:
        return self._classifier.predict(X)

    def predict_proba(self, X: Any) -> Any:
        return self._classifier.predict_proba(X)
