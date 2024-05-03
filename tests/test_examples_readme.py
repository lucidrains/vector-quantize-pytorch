import pytest
from pytest_examples import find_examples, CodeExample, EvalExample


@pytest.mark.parametrize('example', find_examples('README.md'), ids=str)
def test_docstrings(example: CodeExample, eval_example: EvalExample):
    """Test all examples (automatically) found in README.
    
    Usage, in an activated virtual env:
    ```py
    (.venv) pytest tests/test_examples_readme.py
    ```
    
    for a simple check on running the examples, and 
    ```py
    (.venv) pytest tests/test_examples_readme.py --update-examples
    ```

    to lint and format the code in the README.

    """
    if eval_example.update_examples:
        eval_example.format(example)
        eval_example.lint(example)
        eval_example.run_print_check(example)
    else:
        # eval_example.format(example)
        # eval_example.lint(example)
        eval_example.run_print_check(example)
