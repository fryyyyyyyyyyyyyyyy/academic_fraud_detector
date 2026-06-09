"""Tests for text similarity tools."""

import json
import pytest

# Try to import sentence-transformers; skip semantic tests if not available
try:
    import sentence_transformers  # noqa: F401
    HAS_SENTENCE_TRANSFORMERS = True
except ImportError:
    HAS_SENTENCE_TRANSFORMERS = False


class TestLexicalPlagiarismCheck:
    """Test lexical (string-level) plagiarism detection.

    Uses the raw implementation function _lexical_plagiarism_impl
    to avoid needing the full CrewAI @tool runtime.
    """

    @staticmethod
    def _run(text_a, text_b, min_match_length=50):
        """Helper to call the implementation directly."""
        from academic_fraud_detector.tools.text_similarity import _lexical_plagiarism_impl
        return json.loads(_lexical_plagiarism_impl(
            text_a=text_a, text_b=text_b, min_match_length=min_match_length,
        ))

    def test_identical_texts(self):
        """Identical texts should have ratio = 1.0."""
        text = "The quick brown fox jumps over the lazy dog. This is a test sentence with enough length."
        result = self._run(text_a=text, text_b=text)
        assert result["overall_lexical_ratio"] == 1.0
        assert result["flagged"]

    def test_completely_different_texts(self):
        """Completely different texts should have low ratio."""
        result = self._run(
            text_a="The mitochondria is the powerhouse of the cell, generating ATP through oxidative phosphorylation.",
            text_b="Computer vision algorithms detect objects in images using convolutional neural networks for feature extraction.",
        )
        assert result["overall_lexical_ratio"] < 0.3, (
            f"Expected very low ratio for unrelated texts, got {result['overall_lexical_ratio']}"
        )
        assert not result["flagged"]

    def test_partially_overlapping_texts(self):
        """Texts with some shared content should have intermediate ratio."""
        shared = "The experimental results demonstrate a significant improvement over baseline methods across multiple metrics."
        text_a = f"{shared} We observed a 25 percent increase in classification accuracy on the held-out test set."
        text_b = f"{shared} However, the improvement was not statistically significant after correction for multiple comparisons."

        result = self._run(text_a=text_a, text_b=text_b)

        # Should find the shared portion
        assert result["overall_lexical_ratio"] > 0.4
        assert result["significant_match_count"] >= 1

    def test_min_match_length_filter(self):
        """Short matches should be filtered by min_match_length."""
        text_a = "the cat sat on the mat outside"
        text_b = "the dog ran on the log outside"  # Only "the", "on the", "outside" are (mostly short) shared

        result = self._run(text_a=text_a, text_b=text_b, min_match_length=50)
        # With min_match_length=50, no individual match should be long enough
        assert result["significant_match_count"] == 0

    def test_case_insensitive(self):
        """Matching should be case-insensitive."""
        result = self._run(
            text_a="THE RESULTS ARE SIGNIFICANT AND DEMONSTRATE CLEAR IMPROVEMENT OVER THE BASELINE.",
            text_b="the results are significant and demonstrate clear improvement over the baseline.",
        )
        assert result["overall_lexical_ratio"] > 0.9


@pytest.mark.skipif(
    not HAS_SENTENCE_TRANSFORMERS,
    reason="sentence-transformers is not installed. Install with: pip install sentence-transformers",
)
class TestSemanticSimilarityTool:
    """Test semantic (embedding-based) similarity detection.

    Requires sentence-transformers to be installed.
    """

    @pytest.fixture
    def tool(self):
        from academic_fraud_detector.tools.text_similarity import SemanticSimilarityTool
        return SemanticSimilarityTool()

    def test_identical_semantics(self, tool):
        """Paraphrased text should have high semantic similarity."""
        text_a = "The researchers found that exercise improves cardiovascular health in older adults."
        text_b = "Scientists discovered that physical activity enhances heart health among elderly populations."

        result = json.loads(tool._run(text_a=text_a, text_b=text_b, threshold=0.70))
        # Paraphrased text should have fairly high semantic similarity
        assert result["overall_similarity"] > 0.5, (
            f"Expected high similarity for paraphrased text, got {result.get('overall_similarity')}"
        )

    def test_different_semantics(self, tool):
        """Unrelated texts should have low semantic similarity."""
        text_a = "Quantum mechanics describes the behavior of subatomic particles and their wave functions."
        text_b = "The Renaissance was a period of great artistic achievement and cultural transformation in Europe."

        result = json.loads(tool._run(text_a=text_a, text_b=text_b, threshold=0.70))
        assert result["overall_similarity"] < 0.6, (
            f"Expected low similarity for unrelated texts, got {result.get('overall_similarity')}"
        )

    def test_short_text_fallback(self, tool):
        """Very short texts should use full-passage fallback."""
        text_a = "Hello world."
        text_b = "Hello world."

        result = json.loads(tool._run(text_a=text_a, text_b=text_b))
        assert result["method"] == "full-passage", (
            f"Expected full-passage for short text, got {result.get('method')}"
        )
