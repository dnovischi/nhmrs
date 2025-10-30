# Contributing to NHMRS

Thank you for your interest in contributing to the NHMRS (Non-Holonomic Mobile Robot Systems) project!

## Getting Started

1. Fork the repository
2. Clone your fork:
   ```bash
   git clone https://github.com/yourusername/nhmrs.git
   cd nhmrs
   ```
3. Create a virtual environment:
   ```bash
   python -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   ```
4. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Development Workflow

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and test them:
   ```bash
   python demo.py
   ```

3. Commit your changes:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

4. Push to your fork and create a pull request:
   ```bash
   git push origin feature/your-feature-name
   ```

## Code Style

- Follow PEP 8 style guidelines
- Use meaningful variable and function names
- Add docstrings to all functions and classes
- Keep functions focused and modular

## Testing

Before submitting a pull request, ensure:
- The demo script runs without errors
- Your code includes appropriate documentation
- New features include usage examples

## Reporting Issues

When reporting issues, please include:
- Python version
- Operating system
- Steps to reproduce the issue
- Expected vs actual behavior
- Error messages (if any)

## Questions?

Feel free to open an issue for questions or discussions about the project.
