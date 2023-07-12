"""Tests basic properties of the Analysis class."""


def test_import_analysis():
    """Test importing analysis."""
    from pylars import Analysis

    assert Analysis is not None
