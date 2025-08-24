"""Domain-specific constraints for text generation."""

import re
from collections.abc import Callable

import torch

from symbiont.domains.base import (
    FunctionalConstraint,
    PatternConstraint,
    RangeConstraint,
)


class WordCount(RangeConstraint):
    """Constraint on number of words in generated text."""

    def __init__(
        self,
        min_words: int | None = None,
        max_words: int | None = None,
        exactly: int | None = None,
    ):
        """
        Args:
            min_words: Minimum number of words
            max_words: Maximum number of words
            exactly: Exact number of words (overrides min/max)
        """
        super().__init__(min_words, max_words, exactly)

    def _compute_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute word counts from text tensor."""
        texts = self._tensor_to_text(x)
        word_counts = []

        for text in texts:
            # Simple word counting (split on whitespace)
            words = text.strip().split()
            word_counts.append(len(words))

        return torch.tensor(word_counts, device=x.device, dtype=torch.float32)

    def _tensor_to_text(self, x: torch.Tensor) -> list[str]:
        """Convert tensor to text strings."""
        texts = []

        if x.dim() == 1:
            x = x.unsqueeze(0)

        for seq in x:
            # Convert indices to characters (assuming ASCII encoding)
            try:
                chars = [chr(int(idx.item())) for idx in seq if 0 <= idx.item() < 128]
                texts.append("".join(chars))
            except (ValueError, OverflowError):
                texts.append("")  # Fallback for invalid characters

        return texts

    def __repr__(self) -> str:
        if self.min_value == self.max_value and self.min_value is not None:
            return f"WordCount(exactly={int(self.min_value)})"
        return f"WordCount(min={self.min_value}, max={self.max_value})"


class SentenceCount(RangeConstraint):
    """Constraint on number of sentences in generated text."""

    def __init__(
        self,
        min_sentences: int | None = None,
        max_sentences: int | None = None,
        exactly: int | None = None,
    ):
        super().__init__(min_sentences, max_sentences, exactly)

    def _compute_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute sentence counts from text tensor."""
        texts = self._tensor_to_text(x)
        sentence_counts = []

        for text in texts:
            # Count sentences by looking for sentence-ending punctuation
            sentence_endings = re.findall(r"[.!?]+", text)
            sentence_counts.append(len(sentence_endings))

        return torch.tensor(sentence_counts, device=x.device, dtype=torch.float32)

    def _tensor_to_text(self, x: torch.Tensor) -> list[str]:
        """Convert tensor to text strings."""
        texts = []

        if x.dim() == 1:
            x = x.unsqueeze(0)

        for seq in x:
            try:
                chars = [chr(int(idx.item())) for idx in seq if 0 <= idx.item() < 128]
                texts.append("".join(chars))
            except (ValueError, OverflowError):
                texts.append("")

        return texts

    def __repr__(self) -> str:
        if self.min_value == self.max_value and self.min_value is not None:
            return f"SentenceCount(exactly={int(self.min_value)})"
        return f"SentenceCount(min={self.min_value}, max={self.max_value})"


class ContainsWords(PatternConstraint):
    """Constraint requiring text to contain specific words or phrases."""

    def __init__(self, words: str | list[str], case_sensitive: bool = False):
        """
        Args:
            words: Word(s) or phrase(s) to require
            case_sensitive: Whether matching should be case sensitive
        """
        if isinstance(words, str):
            words = [words]

        self.required_words = words
        # Create regex pattern for word boundaries
        patterns = []
        for word in words:
            # Escape special regex characters
            escaped_word = re.escape(word)
            patterns.append(f"\\b{escaped_word}\\b")

        # All words must be present
        full_pattern = "(?=.*" + ")(?=.*".join(patterns) + ")"
        super().__init__(full_pattern, case_sensitive)

    def _sequence_to_string(self, x: torch.Tensor) -> list[str]:
        """Convert tensor to text strings."""
        texts = []

        if x.dim() == 1:
            x = x.unsqueeze(0)

        for seq in x:
            try:
                chars = [chr(int(idx.item())) for idx in seq if 0 <= idx.item() < 128]
                texts.append("".join(chars))
            except (ValueError, OverflowError):
                texts.append("")

        return texts

    def __repr__(self) -> str:
        return f"ContainsWords({self.required_words})"


class AvoidWords(PatternConstraint):
    """Constraint forbidding specific words or phrases."""

    def __init__(self, words: str | list[str], case_sensitive: bool = False):
        """
        Args:
            words: Word(s) or phrase(s) to forbid
            case_sensitive: Whether matching should be case sensitive
        """
        if isinstance(words, str):
            words = [words]

        self.forbidden_words = words
        # Create regex pattern for any forbidden word
        patterns = []
        for word in words:
            escaped_word = re.escape(word)
            patterns.append(f"\\b{escaped_word}\\b")

        # Match if any forbidden word is present
        full_pattern = "|".join(patterns)
        super().__init__(full_pattern, case_sensitive)

    def _sequence_to_string(self, x: torch.Tensor) -> list[str]:
        """Convert tensor to text strings."""
        texts = []

        if x.dim() == 1:
            x = x.unsqueeze(0)

        for seq in x:
            try:
                chars = [chr(int(idx.item())) for idx in seq if 0 <= idx.item() < 128]
                texts.append("".join(chars))
            except (ValueError, OverflowError):
                texts.append("")

        return texts

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Return 1 - (presence of forbidden words)."""
        base_satisfaction = super().satisfaction(x)
        # Invert because we want to forbid these words
        return (
            torch.tensor(
                1.0, device=base_satisfaction.device, dtype=base_satisfaction.dtype
            )
            - base_satisfaction
        )

    def __repr__(self) -> str:
        return f"AvoidWords({self.forbidden_words})"


class MaxReadingLevel(RangeConstraint):
    """Constraint on text reading level/complexity."""

    def __init__(self, max_level: float):
        """
        Args:
            max_level: Maximum allowed reading level (Flesch-Kincaid grade)
        """
        super().__init__(max_value=max_level)
        self.max_level = max_level

    def _compute_value(self, x: torch.Tensor) -> torch.Tensor:
        """Compute reading level using simplified Flesch-Kincaid."""
        texts = self._tensor_to_text(x)
        levels = []

        for text in texts:
            level = self._flesch_kincaid_grade(text)
            levels.append(level)

        return torch.tensor(levels, device=x.device, dtype=torch.float32)

    def _flesch_kincaid_grade(self, text: str) -> float:
        """Simplified Flesch-Kincaid grade level calculation."""
        if not text.strip():
            return 0.0

        # Count sentences
        sentences = len(re.findall(r"[.!?]+", text))
        if sentences == 0:
            sentences = 1

        # Count words
        words = len(text.split())
        if words == 0:
            return 0.0

        # Count syllables (simplified: count vowel groups)
        syllables = len(re.findall(r"[aeiouAEIOU]+", text))
        if syllables == 0:
            syllables = words  # Fallback assumption

        # Flesch-Kincaid formula
        grade = (0.39 * (words / sentences)) + (11.8 * (syllables / words)) - 15.59
        return max(0.0, grade)

    def _tensor_to_text(self, x: torch.Tensor) -> list[str]:
        """Convert tensor to text strings."""
        texts = []

        if x.dim() == 1:
            x = x.unsqueeze(0)

        for seq in x:
            try:
                chars = [chr(int(idx.item())) for idx in seq if 0 <= idx.item() < 128]
                texts.append("".join(chars))
            except (ValueError, OverflowError):
                texts.append("")

        return texts

    def __repr__(self) -> str:
        return f"MaxReadingLevel({self.max_level})"


class Sentiment(FunctionalConstraint):
    """Constraint on text sentiment (requires sentiment model)."""

    def __init__(
        self,
        target_sentiment: str,
        confidence_threshold: float = 0.6,
        sentiment_fn: Callable[[str], tuple[str, float]] | None = None,
    ):
        """
        Args:
            target_sentiment: Target sentiment ("positive", "negative", "neutral")
            confidence_threshold: Minimum confidence for sentiment prediction
            sentiment_fn: Optional custom sentiment analysis function
        """
        if target_sentiment not in ["positive", "negative", "neutral"]:
            raise ValueError("Sentiment must be 'positive', 'negative', or 'neutral'")

        super().__init__(target_sentiment, confidence_threshold)
        self.target_sentiment = target_sentiment
        self.sentiment_fn = sentiment_fn or self._simple_sentiment

    def _predict_function(self, x: torch.Tensor) -> torch.Tensor:
        """Predict sentiment for text sequences."""
        texts = self._tensor_to_text(x)
        predictions = []

        for text in texts:
            sentiment, confidence = self.sentiment_fn(text)

            # Convert to satisfaction score
            if sentiment == self.target_sentiment:
                predictions.append(confidence)
            else:
                predictions.append(1.0 - confidence)

        return torch.tensor(predictions, device=x.device, dtype=torch.float32)

    def _simple_sentiment(self, text: str) -> tuple[str, float]:
        """
        Simple rule-based sentiment analysis.

        In practice, this would use a trained model like BERT or similar.
        """
        positive_words = [
            "good",
            "great",
            "excellent",
            "amazing",
            "wonderful",
            "love",
            "happy",
        ]
        negative_words = [
            "bad",
            "terrible",
            "awful",
            "hate",
            "sad",
            "angry",
            "horrible",
        ]

        text_lower = text.lower()
        pos_count = sum(1 for word in positive_words if word in text_lower)
        neg_count = sum(1 for word in negative_words if word in text_lower)

        if pos_count > neg_count:
            confidence = min(0.9, 0.5 + 0.1 * (pos_count - neg_count))
            return "positive", confidence
        elif neg_count > pos_count:
            confidence = min(0.9, 0.5 + 0.1 * (neg_count - pos_count))
            return "negative", confidence
        else:
            return "neutral", 0.5

    def _tensor_to_text(self, x: torch.Tensor) -> list[str]:
        """Convert tensor to text strings."""
        texts = []

        if x.dim() == 1:
            x = x.unsqueeze(0)

        for seq in x:
            try:
                chars = [chr(int(idx.item())) for idx in seq if 0 <= idx.item() < 128]
                texts.append("".join(chars))
            except (ValueError, OverflowError):
                texts.append("")

        return texts

    def __repr__(self) -> str:
        return f"Sentiment('{self.target_sentiment}', threshold={self.threshold})"


class ParagraphStructure(PatternConstraint):
    """Constraint on paragraph structure and formatting."""

    def __init__(self, min_paragraphs: int = 1, max_paragraphs: int | None = None):
        """
        Args:
            min_paragraphs: Minimum number of paragraphs
            max_paragraphs: Maximum number of paragraphs (None for no limit)
        """
        self.min_paragraphs = min_paragraphs
        self.max_paragraphs = max_paragraphs

        # Pattern matches paragraph breaks (double newlines)
        super().__init__(r"\n\s*\n", case_sensitive=True)

    def _sequence_to_string(self, x: torch.Tensor) -> list[str]:
        """Convert tensor to text strings."""
        texts = []

        if x.dim() == 1:
            x = x.unsqueeze(0)

        for seq in x:
            try:
                chars = [chr(int(idx.item())) for idx in seq if 0 <= idx.item() < 128]
                texts.append("".join(chars))
            except (ValueError, OverflowError):
                texts.append("")

        return texts

    def satisfaction(self, x: torch.Tensor) -> torch.Tensor:
        """Evaluate paragraph structure satisfaction."""
        texts = self._sequence_to_string(x)
        satisfactions = []

        for text in texts:
            # Count paragraph breaks
            paragraph_breaks = len(self.pattern.findall(text))
            paragraph_count = paragraph_breaks + 1  # One more paragraph than breaks

            # Check if within bounds
            if paragraph_count < self.min_paragraphs:
                satisfaction = paragraph_count / self.min_paragraphs
            elif self.max_paragraphs and paragraph_count > self.max_paragraphs:
                satisfaction = self.max_paragraphs / paragraph_count
            else:
                satisfaction = 1.0

            satisfactions.append(satisfaction)

        return torch.tensor(satisfactions, device=x.device, dtype=torch.float32)

    def __repr__(self) -> str:
        if self.max_paragraphs:
            return f"ParagraphStructure(min={self.min_paragraphs}, max={self.max_paragraphs})"
        return f"ParagraphStructure(min={self.min_paragraphs})"
