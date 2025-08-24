"""Tests for core constraint functionality."""

import pytest
import torch

from symbiont.core.constraints import (
    AlwaysFalse,
    AlwaysTrue,
    AndConstraint,
    NotConstraint,
    OrConstraint,
    WeightedConstraint,
)
from symbiont.core.operators import AtLeast, AtMost, Exactly, all_of, any_of, none_of
from symbiont.domains.sequence import Contains, GCContent, Length


class TestBasicConstraints:
    """Test basic constraint functionality."""

    def test_always_true(self):
        """Test AlwaysTrue constraint."""
        constraint = AlwaysTrue()
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        scores = constraint.satisfaction(x)
        assert torch.all(scores == 1.0)

    def test_always_false(self):
        """Test AlwaysFalse constraint."""
        constraint = AlwaysFalse()
        x = torch.tensor([[1, 2, 3], [4, 5, 6]])
        scores = constraint.satisfaction(x)
        assert torch.all(scores == 0.0)

    def test_weighted_constraint(self):
        """Test WeightedConstraint wrapper."""
        base_constraint = AlwaysTrue()
        weighted = WeightedConstraint(base_constraint, weight=0.5)

        x = torch.tensor([[1, 2, 3]])
        scores = weighted.satisfaction(x)
        assert torch.allclose(scores, torch.tensor([0.5]))

    def test_weighted_constraint_clamping(self):
        """Test that WeightedConstraint clamps to [0,1]."""
        base_constraint = AlwaysTrue()
        weighted = WeightedConstraint(base_constraint, weight=2.0)

        x = torch.tensor([[1, 2, 3]])
        scores = weighted.satisfaction(x)
        assert torch.allclose(scores, torch.tensor([1.0]))  # Should clamp to 1.0

    def test_negative_weight_raises_error(self):
        """Test that negative weights raise ValueError."""
        base_constraint = AlwaysTrue()
        with pytest.raises(ValueError, match="Weight must be non-negative"):
            WeightedConstraint(base_constraint, weight=-1.0)


class TestLogicalOperators:
    """Test logical constraint composition."""

    def test_and_constraint(self):
        """Test AND constraint composition."""
        c1 = AlwaysTrue()
        c2 = AlwaysTrue()
        and_constraint = c1 & c2

        assert isinstance(and_constraint, AndConstraint)

        x = torch.tensor([[1, 2, 3]])
        scores = and_constraint.satisfaction(x)
        assert torch.allclose(scores, torch.tensor([1.0]))

    def test_or_constraint(self):
        """Test OR constraint composition."""
        c1 = AlwaysFalse()
        c2 = AlwaysTrue()
        or_constraint = c1 | c2

        assert isinstance(or_constraint, OrConstraint)

        x = torch.tensor([[1, 2, 3]])
        scores = or_constraint.satisfaction(x)
        assert torch.allclose(scores, torch.tensor([1.0]))

    def test_not_constraint(self):
        """Test NOT constraint."""
        c = AlwaysTrue()
        not_constraint = ~c

        assert isinstance(not_constraint, NotConstraint)

        x = torch.tensor([[1, 2, 3]])
        scores = not_constraint.satisfaction(x)
        assert torch.allclose(scores, torch.tensor([0.0]))

    def test_complex_logical_expression(self):
        """Test complex logical combinations."""
        c1 = AlwaysTrue()
        c2 = AlwaysFalse()
        c3 = AlwaysTrue()

        # (c1 & c2) | c3 should be True | False = True
        complex_constraint = (c1 & c2) | c3

        x = torch.tensor([[1, 2, 3]])
        scores = complex_constraint.satisfaction(x)
        assert torch.allclose(scores, torch.tensor([1.0]))


class TestConstraintOperators:
    """Test constraint operator functions."""

    def test_all_of(self):
        """Test all_of operator."""
        constraints = [AlwaysTrue(), AlwaysTrue(), AlwaysTrue()]
        combined = all_of(*constraints)

        x = torch.tensor([[1, 2, 3]])
        scores = combined.satisfaction(x)
        assert torch.allclose(scores, torch.tensor([1.0]))

    def test_any_of(self):
        """Test any_of operator."""
        constraints = [AlwaysFalse(), AlwaysFalse(), AlwaysTrue()]
        combined = any_of(*constraints)

        x = torch.tensor([[1, 2, 3]])
        scores = combined.satisfaction(x)
        assert torch.allclose(scores, torch.tensor([1.0]))

    def test_none_of(self):
        """Test none_of operator."""
        constraints = [AlwaysFalse(), AlwaysFalse()]
        combined = none_of(*constraints)

        x = torch.tensor([[1, 2, 3]])
        scores = combined.satisfaction(x)
        assert torch.allclose(scores, torch.tensor([1.0]))

    def test_at_least(self):
        """Test AtLeast constraint."""
        constraints = [AlwaysTrue(), AlwaysTrue(), AlwaysFalse()]
        at_least_2 = AtLeast(2, *constraints)

        x = torch.tensor([[1, 2, 3]])
        scores = at_least_2.satisfaction(x)
        assert torch.allclose(scores, torch.tensor([1.0]))  # 2 out of 3 satisfied

    def test_at_most(self):
        """Test AtMost constraint."""
        constraints = [AlwaysTrue(), AlwaysFalse(), AlwaysFalse()]
        at_most_1 = AtMost(1, *constraints)

        x = torch.tensor([[1, 2, 3]])
        scores = at_most_1.satisfaction(x)
        assert torch.allclose(scores, torch.tensor([1.0]))  # 1 out of 3 satisfied

    def test_exactly(self):
        """Test Exactly constraint."""
        constraints = [AlwaysTrue(), AlwaysTrue(), AlwaysFalse()]
        exactly_2 = Exactly(2, *constraints)

        x = torch.tensor([[1, 2, 3]])
        scores = exactly_2.satisfaction(x)
        assert torch.allclose(scores, torch.tensor([1.0]))  # Exactly 2 satisfied

    def test_empty_constraints_raises_error(self):
        """Test that empty constraint lists raise errors."""
        with pytest.raises(ValueError):
            all_of()
        with pytest.raises(ValueError):
            any_of()
        with pytest.raises(ValueError):
            none_of()


class TestSequenceConstraints:
    """Test sequence-specific constraints."""

    def test_contains_constraint(self):
        """Test Contains constraint with DNA sequences."""
        constraint = Contains("ATG")

        # Mock DNA sequences: 0=A, 1=T, 2=G, 3=C
        # ATG = [0, 1, 2]
        seq_with_atg = torch.tensor([[0, 1, 2, 3, 0, 1]])  # ATGCCA (has ATG)
        seq_without_atg = torch.tensor([[0, 0, 3, 3, 1, 1]])  # AACCTT (no ATG)

        scores_with = constraint.satisfaction(seq_with_atg)
        scores_without = constraint.satisfaction(seq_without_atg)

        assert scores_with.item() == 1.0
        assert scores_without.item() == 0.0

    def test_length_constraint(self):
        """Test Length constraint."""
        constraint = Length(5, 10)  # Length between 5-10

        short_seq = torch.tensor([[1, 2, 3]])  # Length 3 (too short)
        good_seq = torch.tensor([[1, 2, 3, 4, 5, 6]])  # Length 6 (good)
        long_seq = torch.tensor(
            [[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]]
        )  # Length 12 (too long)

        short_score = constraint.satisfaction(short_seq).item()
        good_score = constraint.satisfaction(good_seq).item()
        long_score = constraint.satisfaction(long_seq).item()

        assert good_score > short_score
        assert good_score > long_score
        assert good_score > 0.9  # Should be high satisfaction

    def test_gc_content_constraint(self):
        """Test GC content constraint."""
        constraint = GCContent(0.4, 0.6)  # 40-60% GC

        # All A,T (0% GC): [0,0,1,1,1,1] = AATTTT
        low_gc = torch.tensor([[0, 0, 1, 1, 1, 1]])

        # 50% GC: [0,1,2,3,2,3] = ATGCGC
        good_gc = torch.tensor([[0, 1, 2, 3, 2, 3]])

        # All G,C (100% GC): [2,2,3,3,2,3] = GGCCGC
        high_gc = torch.tensor([[2, 2, 3, 3, 2, 3]])

        low_score = constraint.satisfaction(low_gc).item()
        good_score = constraint.satisfaction(good_gc).item()
        high_score = constraint.satisfaction(high_gc).item()

        assert good_score > low_score
        assert good_score > high_score

    def test_batch_constraints(self):
        """Test constraints with batched sequences."""
        constraint = Length(4, 6)

        # Test with sequences of different actual lengths
        good_seq = torch.tensor([[1, 2, 3, 4, 5]])  # Length 5 (good)
        short_seq = torch.tensor([[1, 2, 3]])  # Length 3 (too short)
        long_seq = torch.tensor([[1, 2, 3, 4, 5, 6, 7]])  # Length 7 (too long)

        good_score = constraint.satisfaction(good_seq).item()
        short_score = constraint.satisfaction(short_seq).item()
        long_score = constraint.satisfaction(long_seq).item()

        assert good_score > short_score  # Good length > short length
        assert good_score > long_score  # Good length > long length
        assert good_score > 0.9  # Should have high satisfaction


class TestConstraintEdgeCases:
    """Test edge cases and error conditions."""

    def test_empty_sequence(self):
        """Test constraints with empty sequences."""
        constraint = Length(1, 10)
        empty_seq = torch.tensor([[]])  # Empty sequence

        scores = constraint.satisfaction(empty_seq)
        assert scores.item() < 0.5  # Should have low satisfaction for empty sequence

    def test_single_element_sequence(self):
        """Test constraints with single-element sequences."""
        constraint = Contains("A")
        single_seq = torch.tensor([[0]])  # Single 'A'

        scores = constraint.satisfaction(single_seq)
        assert scores.item() == 1.0  # Should find the 'A'

    def test_constraint_with_invalid_parameters(self):
        """Test constraints with invalid parameters."""
        with pytest.raises(ValueError):
            Length(10, 5)  # Max < Min should raise error

        with pytest.raises(ValueError):
            GCContent(-0.1, 0.5)  # Invalid GC ratio

        with pytest.raises(ValueError):
            GCContent(0.8, 0.3)  # Min > Max should raise error

    def test_constraint_repr(self):
        """Test constraint string representations."""
        c1 = Length(5, 10)
        c2 = Contains("ATG")
        c3 = GCContent(0.4, 0.6)

        assert "Length" in str(c1)
        assert "Contains" in str(c2)
        assert "GCContent" in str(c3)

        # Test composite constraint repr
        composite = c1 & c2
        assert "AndConstraint" in str(composite)


if __name__ == "__main__":
    pytest.main([__file__])
