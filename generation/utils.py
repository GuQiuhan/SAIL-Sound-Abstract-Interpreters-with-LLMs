def combine_remove_comments(prmpt, cmpln, old) -> str:
    """Combine and remove comments from the given Python code."""
    cmpln = re.sub(r"'''(.*?)'''", "", cmpln, flags=re.DOTALL)
    cmpln = re.sub(r'"""(.*?)"""', "", cmpln, flags=re.DOTALL)
    old = "\n".join(
        line
        for line in old.splitlines()
        if "```python" not in line and "```" not in line
    )
    cmpln = "\n".join(
        line
        for line in cmpln.splitlines()
        if "```python" not in line and "```" not in line
    )

    return old + "\n\n" + cmpln
    # return cmpln

def remove_comments(prmpt, cmpln, old) -> str:
    """Remove comments from the given Python code."""
    cmpln = re.sub(r"'''(.*?)'''", "", cmpln, flags=re.DOTALL)
    cmpln = re.sub(r'"""(.*?)"""', "", cmpln, flags=re.DOTALL)

    cmpln = "\n".join(
        line
        for line in cmpln.splitlines()
        if "```python" not in line and "```" not in line
    )

    return cmpln