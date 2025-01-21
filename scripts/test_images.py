import subprocess


def handle_error(test_name, stdout, stderr, exception=None):
    """
    Centralized error handling function for tests.
    Args:
        test_name (str): Name of the test that failed.
        stdout (str): Standard output of the failed command.
        stderr (str): Standard error of the failed command.
        exception (Exception, optional): Exception object to include details. Defaults to None.
    """
    print(f"[TEST FAILED] {test_name}")
    if stdout.strip():
        print("STDOUT:")
        print(stdout.strip())
    if stderr.strip():
        print("STDERR:")
        print(stderr.strip())
    if exception:
        print(f"Details: {exception}")


def run_script(args):
    """
    Run the Docker management script with the given arguments and return the output.
    Args:
        args (list): Arguments to pass to the script.
    Returns:
        tuple: stdout, stderr, and returncode of the executed script.
    """
    result = subprocess.run(
        ["python", "images.py"] + args,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    return result.stdout, result.stderr, result.returncode


def validate_output(output, expected_phrases):
    """
    Validate that all lines in the output contain at least one expected phrase.
    Args:
        output (str): Output to validate.
        expected_phrases (list): List of expected phrases.
    Returns:
        bool: True if all lines match the expected phrases, False otherwise.
    """
    for line in output.splitlines():
        if not any(phrase in line for phrase in expected_phrases):
            return False
    return True


def test_build():
    """Test the build functionality."""
    try:
        stdout, stderr, returncode = run_script(["--mode", "build", "--tag", "1.0.0"])
        assert returncode == 0, f"Build failed: {stderr}"
        assert "[OK] Built image" in stdout, f"Unexpected output: {stdout}"
        print("[TEST PASSED] Build functionality works correctly.")
    except AssertionError as e:
        handle_error("Build functionality", stdout, stderr, e)


def test_remove():
    """Test the remove functionality."""
    try:
        stdout, stderr, returncode = run_script(["--mode", "remove"])
        assert returncode == 0, f"Remove failed: {stderr}"
        assert validate_output(
            stdout,
            expected_phrases=["[INFO] Image not found, skipping:", "[OK] Removed image:"],
        ), f"Unexpected output: {stdout}"
        print("[TEST PASSED] Remove functionality works correctly.")
    except AssertionError as e:
        handle_error("Remove functionality", stdout, stderr, e)


def test_tag():
    """Test the tag functionality."""
    try:
        stdout, stderr, returncode = run_script(
            ["--mode", "tag", "--old-tag", "1.0.0", "--new-tag", "latest"]
        )
        assert returncode == 0, f"Tag failed: {stderr}"
        assert validate_output(
            stdout,
            expected_phrases=["[OK] Tagged image:"],
        ), f"Unexpected output: {stdout}"
        print("[TEST PASSED] Tag functionality works correctly.")
    except AssertionError as e:
        handle_error("Tag functionality", stdout, stderr, e)


def test_unsupported_mode():
    """Test unsupported mode handling."""
    try:
        stdout, stderr, returncode = run_script(["--mode", "invalid_mode"])
        assert returncode != 0, "Unsupported mode did not fail as expected."
        print("[TEST PASSED] Unsupported mode handling works correctly.")
    except AssertionError as e:
        handle_error("Unsupported mode handling", stdout, stderr, e)


if __name__ == "__main__":
    test_build()
    test_remove()
    test_tag()
    test_unsupported_mode()