import sys

import src


def test_python_version():
    """Ensure tests are running with a compatible Python version."""
    assert sys.version_info >= (3, 8), "Python version must be 3.8 or higher."


def test_environment_setup():
    """Test if the src directory is correctly accessible."""

    assert src is not None
