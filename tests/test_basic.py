"""
Basic Sanity & Structure Tests for KisanMitra AI.
Compatible with both pytest and python standard library unittest.
"""

from pathlib import Path
import unittest


class TestProjectStructure(unittest.TestCase):
    """Verifies that all core folders, placeholder files, and configs exist."""

    @classmethod
    def setUpClass(cls):
        cls.base_dir = Path(__file__).resolve().parent.parent

    def test_required_directories_exist(self):
        """Verify that all core project directories exist."""
        required_dirs = [
            self.base_dir / "data" / "raw",
            self.base_dir / "data" / "processed",
            self.base_dir / "notebooks",
            self.base_dir / "src" / "disease",
            self.base_dir / "src" / "price",
            self.base_dir / "src" / "utils",
            self.base_dir / "models",
            self.base_dir / "app",
            self.base_dir / "reports",
            self.base_dir / "tests",
        ]

        for directory in required_dirs:
            self.assertTrue(
                directory.exists(), f"Missing required directory: {directory}"
            )
            self.assertTrue(
                directory.is_dir(), f"Path is not a directory: {directory}"
            )

    def test_required_root_files_exist(self):
        """Verify that root configuration and documentation files exist."""
        required_files = [
            self.base_dir / "README.md",
            self.base_dir / "requirements.txt",
            self.base_dir / ".gitignore",
            self.base_dir / ".env.example",
            self.base_dir / "reports" / "results.md",
        ]

        for file_path in required_files:
            self.assertTrue(
                file_path.exists(), f"Missing required file: {file_path}"
            )
            self.assertTrue(
                file_path.is_file(), f"Path is not a file: {file_path}"
            )

    def test_package_import(self):
        """Verify that the src package can be imported and has version metadata."""
        import sys
        if str(self.base_dir) not in sys.path:
            sys.path.insert(0, str(self.base_dir))

        import src
        self.assertEqual(src.__version__, "0.1.0")


if __name__ == "__main__":
    unittest.main()
