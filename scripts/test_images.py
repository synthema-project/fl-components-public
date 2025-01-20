import subprocess
import os
from pathlib import Path

def run_script(args):
    """Run the script with given arguments and return the output."""
    result = subprocess.run(
        ["python", "images.py"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True
    )
    return result.stdout, result.stderr, result.returncode

def test_build():
    """Test the build functionality."""
    try:
        stdout, stderr, returncode = run_script(["--mode", "build", "--tag", "1.0.0"])
        assert returncode == 0, f"Build failed: {stderr}"
        assert "[OK] Built image" in stdout, f"Unexpected output: {stdout}"
        print("[TEST PASSED] Build functionality works correctly.")
    except AssertionError as e:
        print(f"[TEST FAILED] {e}")

def test_remove():
    """Test the remove functionality."""
    try:
        stdout, stderr, returncode = run_script(["--mode", "remove"])
        assert returncode == 0, f"Remove failed: {stderr}"
        assert "[INFO] Removing images..." in stdout, f"Unexpected output: {stdout}"
        print("[TEST PASSED] Remove functionality works correctly.")
    except AssertionError as e:
        print(f"[TEST FAILED] {e}")

def test_tag():
    """Test the tag functionality."""
    try:
        stdout, stderr, returncode = run_script(["--mode", "tag", "--old-tag", "1.0.0", "--new-tag", "latest"])
        assert returncode == 0, f"Tag failed: {stderr}"
        assert "[INFO] Tagging images..." in stdout, f"Unexpected output: {stdout}"
        print("[TEST PASSED] Tag functionality works correctly.")
    except AssertionError as e:
        print(f"[TEST FAILED] {e}")

def test_unsupported_mode():
    """Test unsupported mode handling."""
    try:
        stdout, stderr, returncode = run_script(["--mode", "invalid_mode"])
        assert returncode != 0, "Unsupported mode did not fail as expected."
        print("[TEST PASSED] Unsupported mode handling works correctly.")
    except AssertionError as e:
        print(f"[TEST FAILED] {e}")

if __name__ == "__main__":
    test_build()
    test_remove()
    test_tag()
    test_unsupported_mode()