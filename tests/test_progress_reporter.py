"""Tests for ProgressReporter visualization utility."""

import io
from unittest.mock import patch

import pytest

from symbiont.core.constraints import AlwaysTrue
from symbiont.domains.sequence import Contains, GCContent, Length
from symbiont.generators.mock import MockSequenceGenerator
from symbiont.utils.visualization import ConstraintProgress, ProgressReporter


class TestConstraintProgress:
    """Test ConstraintProgress dataclass."""

    def test_initialization(self):
        """Test basic initialization."""
        progress = ConstraintProgress("test_constraint")
        assert progress.name == "test_constraint"
        assert progress.current_score == 0.0
        assert progress.initial_score == 0.0
        assert progress.best_score == 0.0
        assert len(progress.history) == 0
        assert progress.target_threshold == 0.8

    def test_update(self):
        """Test score updating."""
        progress = ConstraintProgress("test")

        # Update with improving scores
        progress.update(0.3)
        assert progress.current_score == 0.3
        assert progress.best_score == 0.3
        assert len(progress.history) == 1

        progress.update(0.5)
        assert progress.current_score == 0.5
        assert progress.best_score == 0.5
        assert len(progress.history) == 2

        # Update with worse score
        progress.update(0.4)
        assert progress.current_score == 0.4
        assert progress.best_score == 0.5  # Best score unchanged
        assert len(progress.history) == 3

    def test_improvement_calculation(self):
        """Test improvement metrics."""
        progress = ConstraintProgress("test")
        progress.initial_score = 0.3
        progress.current_score = 0.7

        assert progress.improvement == pytest.approx(0.4)

    def test_improvement_rate(self):
        """Test improvement rate calculation."""
        progress = ConstraintProgress("test")

        # No history - should return 0
        assert progress.improvement_rate == 0.0

        # Single entry - should return 0
        progress.update(0.3)
        assert progress.improvement_rate == 0.0

        # Multiple entries
        for score in [0.4, 0.5, 0.6, 0.7]:
            progress.update(score)

        # Should calculate rate over recent history
        rate = progress.improvement_rate
        assert rate > 0  # Positive improvement


class TestProgressReporter:
    """Test ProgressReporter class."""

    def test_initialization(self):
        """Test reporter initialization."""
        constraints = [AlwaysTrue(), GCContent(0.4, 0.6)]
        names = ["Always True", "GC Content"]

        reporter = ProgressReporter(
            constraints,
            constraint_names=names,
            target_threshold=0.75,
            update_interval=0.05,
        )

        assert len(reporter.constraints) == 2
        assert len(reporter.progress_trackers) == 2
        assert "Always True" in reporter.progress_trackers
        assert "GC Content" in reporter.progress_trackers
        assert reporter.target_threshold == 0.75
        assert reporter.update_interval == 0.05

    def test_default_constraint_names(self):
        """Test automatic constraint naming."""
        constraints = [AlwaysTrue(), GCContent(0.4, 0.6), Length(20, 30)]
        reporter = ProgressReporter(constraints)

        assert "Constraint_0" in reporter.progress_trackers
        assert "Constraint_1" in reporter.progress_trackers
        assert "Constraint_2" in reporter.progress_trackers

    def test_initialize_tracking(self):
        """Test initialization with sequences."""
        constraints = [AlwaysTrue(), Contains("ATG")]
        reporter = ProgressReporter(constraints)

        # Generate mock sequences
        generator = MockSequenceGenerator(vocab_size=4, sequence_length=20)
        sequences = generator.generate()

        # Capture output
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            reporter.initialize(sequences, max_iterations=10)

        # Check initialization
        assert reporter.max_iterations == 10
        for tracker in reporter.progress_trackers.values():
            assert tracker.initial_score >= 0  # Score can be 0 for Contains
            assert len(tracker.history) == 1
            assert tracker.initial_score == tracker.current_score

    def test_update_tracking(self):
        """Test updating progress with new sequences."""
        constraints = [AlwaysTrue(), GCContent(0.4, 0.6)]
        reporter = ProgressReporter(constraints, update_interval=0.0)

        generator = MockSequenceGenerator(vocab_size=4, sequence_length=20)
        sequences1 = generator.generate()
        sequences2 = generator.generate()

        # Initialize
        with patch("sys.stdout", io.StringIO()):
            reporter.initialize(sequences1)
            reporter.update(sequences2, iteration=1)

        # Check updates
        assert reporter.iteration == 1
        for tracker in reporter.progress_trackers.values():
            assert len(tracker.history) >= 2

    def test_finalize_summary(self):
        """Test finalization and summary generation."""
        constraints = [AlwaysTrue(), GCContent(0.4, 0.6)]
        names = ["Always", "GC"]
        reporter = ProgressReporter(constraints, constraint_names=names)

        generator = MockSequenceGenerator(vocab_size=4, sequence_length=20)
        sequences = generator.generate()

        # Run tracking
        with patch("sys.stdout", io.StringIO()):
            reporter.initialize(sequences)
            for i in range(3):
                reporter.update(sequences, iteration=i + 1)
            summary = reporter.finalize()

        # Check summary structure
        assert "total_time" in summary
        assert "iterations" in summary
        assert "constraints" in summary
        assert "mean_score" in summary
        assert "success_rate" in summary

        # Check constraint summaries
        assert "Always" in summary["constraints"]
        assert "GC" in summary["constraints"]

        for constraint_summary in summary["constraints"].values():
            assert "final_score" in constraint_summary
            assert "initial_score" in constraint_summary
            assert "best_score" in constraint_summary
            assert "improvement" in constraint_summary
            assert "success" in constraint_summary
            assert "history" in constraint_summary

    def test_color_output_disabled_non_tty(self):
        """Test that color output is disabled for non-TTY."""
        constraints = [AlwaysTrue()]

        # Mock non-TTY environment
        with patch("sys.stdout.isatty", return_value=False):
            reporter = ProgressReporter(constraints, use_color=True)
            assert not reporter.use_color

    def test_render_progress_bar(self):
        """Test progress bar rendering."""
        constraints = [GCContent(0.4, 0.6)]
        reporter = ProgressReporter(constraints, constraint_names=["GC"])

        # Set up a tracker with specific values
        tracker = reporter.progress_trackers["GC"]
        tracker.current_score = 0.75
        tracker.initial_score = 0.5
        tracker.best_score = 0.8
        tracker.history = [0.5, 0.6, 0.7, 0.75]

        # Capture output
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            reporter._render_progress_bar("GC", tracker)

        output = captured_output.getvalue()
        assert "GC" in output
        assert "0.750" in output  # Current score
        assert "+0.250" in output  # Improvement

    def test_rate_limiting(self):
        """Test update rate limiting."""
        constraints = [AlwaysTrue()]
        reporter = ProgressReporter(
            constraints, update_interval=1.0
        )  # 1 second interval

        generator = MockSequenceGenerator(vocab_size=4, sequence_length=20)
        sequences = generator.generate()

        with patch("sys.stdout", io.StringIO()):
            reporter.initialize(sequences)

            # First update should work
            reporter.update(sequences, iteration=1)
            first_update_time = reporter.last_update

            # Immediate second update should be skipped
            reporter.update(sequences, iteration=2)
            assert reporter.iteration == 1  # Iteration shouldn't change
            assert reporter.last_update == first_update_time  # Time shouldn't change

    def test_eta_calculation(self):
        """Test ETA calculation in header."""
        constraints = [AlwaysTrue()]
        reporter = ProgressReporter(constraints, show_eta=True)

        generator = MockSequenceGenerator(vocab_size=4, sequence_length=20)
        sequences = generator.generate()

        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            reporter.initialize(sequences, max_iterations=10)
            reporter.iteration = 5  # Halfway through
            reporter._render()

        output = captured_output.getvalue()
        assert "ETA:" in output or reporter.iteration == 0

    def test_status_indicators(self):
        """Test different status indicators based on progress."""
        constraints = [GCContent(0.4, 0.6)]
        reporter = ProgressReporter(constraints, constraint_names=["GC"])

        tracker = reporter.progress_trackers["GC"]

        # Test success indicator
        tracker.current_score = 0.85
        tracker.target_threshold = 0.8
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            reporter._render_progress_bar("GC", tracker)
        assert "✓" in captured_output.getvalue()

        # Test improving indicator
        tracker.current_score = 0.6
        tracker.history = [0.3, 0.4, 0.5, 0.55, 0.6]  # Improving trend
        captured_output = io.StringIO()
        with patch("sys.stdout", captured_output):
            reporter._render_progress_bar("GC", tracker)
        output = captured_output.getvalue()
        # Should show either success or trend indicator
        assert any(indicator in output for indicator in ["✓", "↑", "↓", "→"])


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
