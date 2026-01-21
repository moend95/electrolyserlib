# ElectrolyserLib - Project Overview

## 📁 Project Structure

```
electrolyserlib/
├── electrolyserlib/          # Main package
│   ├── __init__.py          # Package initialization, exports classes
│   └── pem_electrolyser.py  # Main module with Electrolyser and DynamicElectrolyser
├── examples/                 # Example scripts
│   ├── basic_usage.py       # Basic examples
│   └── dynamic_simulation.py # Dynamic simulation
├── .gitignore               # Git exclusions
├── LICENSE                  # MIT License
├── MANIFEST.in              # Additional files for distribution
├── PUBLISHING.md            # Detailed publishing guide
├── pyproject.toml           # Modern Python package configuration
├── README.md                # Main documentation
└── setup.py                 # Setup script for compatibility
```

## ✅ Status

- ✅ Package structure created
- ✅ Local installation tested (`pip install -e .`)
- ✅ Import works
- ✅ Examples work
- ⏳ Not yet published to PyPI

## 🚀 Next Steps for Publishing

### 1. Customize Personal Information

Edit [pyproject.toml](pyproject.toml):
```toml
[project]
authors = [
    {name = "Your Name", email = "your.email@example.com"}
]
```

### 2. Check if the name "electrolyserlib" is available

Visit: https://pypi.org/project/electrolyserlib/

If the name is taken, change it in `pyproject.toml`:
```toml
name = "electrolyserlib-moend"  # Or another unique name
```

### 3. Install Build Tools

```powershell
pip install build twine
```

### 4. Build Package

```powershell
# Delete old builds
Remove-Item -Recurse -Force dist, build, *.egg-info -ErrorAction SilentlyContinue

# Create new build
python -m build
```

### 5. Test on TestPyPI (recommended)

```powershell
# Create account on https://test.pypi.org
# Then upload:
python -m twine upload --repository testpypi dist/*
```

### 6. Publish to PyPI

```powershell
# Create account on https://pypi.org
# Then upload:
python -m twine upload dist/*
```

## 📖 Documentation

- **README.md**: Comprehensive English documentation with examples
- **PUBLISHING.md**: Detailed step-by-step publishing guide
- **Examples**: Two complete, working example scripts

## 🔧 Local Development

```powershell
# Install in development mode
pip install -e .

# Run tests
python examples/basic_usage.py
python examples/dynamic_simulation.py

# Test import
python -c "from electrolyserlib import Electrolyser, DynamicElectrolyser; print('OK')"
```

## 📦 Features

- **Electrolyser**: Base class for H2 production calculation
- **DynamicElectrolyser**: Advanced class with startup/standby logic
- **Flexible Inputs**: Various units (W, kW, MW) and resolutions (1min - 1h)
- **Default Curve**: Integrated PEM efficiency curve
- **Custom Curves**: Ability to use custom CSV data

## 🎯 Use Cases

- Renewable Energy Integration
- System Optimization
- Energy Storage Analysis
- Feasibility Studies
- Grid Services & Demand Response

## 📝 License

MIT License - Free for commercial and non-commercial use

## 🤝 Contributing

After publishing, others can contribute via GitHub Issues and Pull Requests.

---

**Important**: Read [PUBLISHING.md](PUBLISHING.md) for the detailed step-by-step guide for publishing to PyPI!
