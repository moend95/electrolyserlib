# Publishing Guide for ElectrolyserLib

This guide shows you how to publish your library to PyPI.

## Preparation

### 1. Create PyPI Account

1. Register on [PyPI](https://pypi.org/account/register/)
2. Verify your email address
3. Optional: Also create an account on [TestPyPI](https://test.pypi.org/account/register/) for testing

### 2. Install Packages

```powershell
pip install build twine
```

## Test Local Installation

Before publishing, you should test the package locally:

```powershell
# In the project directory
pip install -e .
```

Then test the installation:

```powershell
python -c "from electrolyserlib import Electrolyser; print('Import successful!')"
```

## Create Build

```powershell
# Delete old builds
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Create new build
python -m build
```

This creates two files in `dist/`:
- `electrolyserlib-0.1.0.tar.gz` (Source Distribution)
- `electrolyserlib-0.1.0-py3-none-any.whl` (Wheel)

## Publish to TestPyPI (recommended for first test)

```powershell
# Upload to TestPyPI
python -m twine upload --repository testpypi dist/*
```

You will be prompted for your TestPyPI username and password.

Test the installation from TestPyPI:

```powershell
pip install --index-url https://test.pypi.org/simple/ --extra-index-url https://pypi.org/simple/ electrolyserlib
```

## Publish to PyPI

If everything works, publish to the real PyPI:

```powershell
python -m twine upload dist/*
```

You will be prompted for your PyPI username and password.

## Use API Token (recommended)

Instead of a password, you can use an API token:

1. Go to [PyPI Account Settings](https://pypi.org/manage/account/)
2. Create an API token
3. Create a `.pypirc` file in your home directory:

```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # Your token here
```

## Installation by Users

After publishing, users can install your library:

```bash
pip install electrolyserlib
```

## Publish Updates

1. Update the version in `pyproject.toml`
2. Create a new build: `python -m build`
3. Upload: `python -m twine upload dist/*`

## Important Notes

- **Versions cannot be overwritten**: Once uploaded, a version cannot be changed
- **Project names are unique**: The name "electrolyserlib" must still be available on PyPI
- **Check before uploading**: Always test on TestPyPI first
- **Documentation**: Keep the README up to date, as it is displayed on PyPI

## Pre-Publication Checklist

- [ ] All tests pass
- [ ] README.md is current and complete
- [ ] Version in pyproject.toml is correct
- [ ] Your contact details in pyproject.toml are updated
- [ ] LICENSE file is present
- [ ] Code is clean and documented
- [ ] Local installation works (`pip install -e .`)
- [ ] TestPyPI upload was successful

## Useful Commands

```powershell
# Check the package before upload
python -m twine check dist/*

# Show package information
pip show electrolyserlib

# Uninstall the package
pip uninstall electrolyserlib
```

## Additional Resources

- [Python Packaging User Guide](https://packaging.python.org/)
- [PyPI Help](https://pypi.org/help/)
- [Twine Documentation](https://twine.readthedocs.io/)
