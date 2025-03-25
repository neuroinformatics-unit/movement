"""Demonstrates best practices for testing error messages in pytest.

This module serves as a guide and example for how to properly test
error messages across the movement codebase.
"""

import re
from contextlib import nullcontext as does_not_raise

import pytest


def example_function_with_error(value):
    """Example function that raises an error with a specific message."""
    if value < 0:
        raise ValueError("Value must be greater than or equal to zero.")
    if value > 100:
        raise ValueError("Value must be less than or equal to 100.")
    if not isinstance(value, int):
        raise TypeError(f"Expected int, got {type(value).__name__}.")
    return value


class TestErrorMessages:
    """Demonstrates various approaches to testing error messages."""

    @pytest.mark.parametrize(
        "value, expected_exception, expected_message",
        [
            # Happy path - no error
            (50, does_not_raise(), None),
            # Testing for specific error messages
            (-10, pytest.raises(ValueError), "greater than or equal to zero"),
            (110, pytest.raises(ValueError), "less than or equal to 100"),
            (1.5, pytest.raises(TypeError), "Expected int, got float"),
        ],
    )
    def test_using_match_parameter(self, value, expected_exception, expected_message):
        """Method 1: Using the match parameter with pytest.raises.
        
        This approach directly uses the match parameter which expects a regex pattern.
        """
        if expected_message:
            with expected_exception as excinfo:
                example_function_with_error(value)
            # Additional validation if needed
            assert expected_message in str(excinfo.value)
        else:
            with expected_exception:
                example_function_with_error(value)

    @pytest.mark.parametrize(
        "value, expected_exception, expected_message",
        [
            # Happy path - no error
            (50, does_not_raise(), None),
            # Testing for specific error messages
            (-10, pytest.raises(ValueError), "greater than or equal to zero"),
            (110, pytest.raises(ValueError), "less than or equal to 100"),
            (1.5, pytest.raises(TypeError), "Expected int, got float"),
        ],
    )
    def test_using_match_parameter_inline(self, value, expected_exception, expected_message):
        """Method 2: Using the match parameter directly in pytest.raises.
        
        This approach combines the context manager creation with pattern matching.
        """
        if expected_message:
            with pytest.raises(expected_exception.expected_exception, match=expected_message):
                example_function_with_error(value)
        else:
            with expected_exception:
                example_function_with_error(value)

    @pytest.mark.parametrize(
        "value, expected_exception, expected_message",
        [
            # Happy path - no error
            (50, does_not_raise(), None),
            # Testing for specific error messages
            (-10, pytest.raises(ValueError), "greater than or equal to zero"),
            (110, pytest.raises(ValueError), "less than or equal to 100"),
            (1.5, pytest.raises(TypeError), "Expected int, got float"),
        ],
    )
    def test_using_helper_function(self, value, expected_exception, expected_message, check_error):
        """Method 3: Using a helper function from conftest.py.
        
        This approach uses a shared helper function to check error messages,
        promoting consistency across the test suite.
        """
        if expected_message:
            with expected_exception as excinfo:
                example_function_with_error(value)
            assert check_error(excinfo, expected_message)
        else:
            with expected_exception:
                example_function_with_error(value)

    def test_with_exact_message_using_re_escape(self):
        """Method 4: When exact message matching is needed.
        
        Use re.escape when you need to match the entire message exactly,
        including any special regex characters.
        """
        exact_message = "Value must be greater than or equal to zero."
        with pytest.raises(ValueError, match=re.escape(exact_message)):
            example_function_with_error(-10)
            

def test_recommended_approach():
    """The recommended approach for testing error messages in movement.
    
    RECOMMENDED APPROACH:
    For most cases, use pytest.raises with the match parameter.
    This is the simplest and most direct method.
    """
    # For simple substring matching
    with pytest.raises(ValueError, match="greater than or equal to zero"):
        example_function_with_error(-10)
    
    # For exact message matching
    exact_message = "Value must be greater than or equal to zero."
    with pytest.raises(ValueError, match=re.escape(exact_message)):
        example_function_with_error(-10)
    
    # For regex pattern matching
    with pytest.raises(TypeError, match=r"Expected int, got \w+\."):
        example_function_with_error(1.5)
        
    # When testing complex cases with multiple assertions
    with pytest.raises(ValueError) as excinfo:
        example_function_with_error(-10)
    assert "greater than" in str(excinfo.value)
    assert "zero" in str(excinfo.value) 