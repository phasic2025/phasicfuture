# Contributing to Topological Neural Network

Thank you for your interest in contributing to this project! This document provides guidelines for contributing to the Topological Neural Network project.

## Getting Started

### Prerequisites

- Julia 1.6 or later
- Git
- Basic understanding of Julia programming
- Familiarity with neural networks and topological data analysis (helpful but not required)

### Setting Up Your Development Environment

1. Fork the repository on GitHub
2. Clone your fork locally:
   ```bash
   git clone https://github.com/yourusername/TopologicalNeuralNetwork.git
   cd TopologicalNeuralNetwork
   ```

3. Add the upstream repository:
   ```bash
   git remote add upstream https://github.com/originalowner/TopologicalNeuralNetwork.git
   ```

4. Install dependencies:
   ```bash
   julia --project=@. -e 'using Pkg; Pkg.instantiate()'
   ```

## Development Workflow

### 1. Create a Branch

Create a new branch for your feature or bug fix:
```bash
git checkout -b feature/your-feature-name
```

### 2. Make Changes

- Follow the existing code style and conventions
- Add comments for complex algorithms
- Update documentation as needed
- Write tests for new functionality

### 3. Test Your Changes

Run the test suite:
```bash
julia --project=@. -e 'using Pkg; Pkg.test()'
```

### 4. Commit Your Changes

Write clear, descriptive commit messages:
```bash
git add .
git commit -m "Add feature: description of your changes"
```

### 5. Push and Create a Pull Request

```bash
git push origin feature/your-feature-name
```

Then create a pull request on GitHub.

## Code Style Guidelines

### Julia Code Style

- Follow the official Julia style guide
- Use 4 spaces for indentation (no tabs)
- Keep lines under 92 characters
- Use descriptive variable and function names
- Add type annotations where appropriate
- Document functions with docstrings

### Documentation

- Update README.md for user-facing changes
- Add inline comments for complex algorithms
- Update relevant documentation in the docs/ directory
- Include examples in docstrings

## Project Structure

```
src/
├── core/           # Core neural network implementations
├── ui/             # User interface components
├── servers/        # Web server implementations
├── demos/          # Demo applications
└── utils/          # Utility functions and tests
```

## Types of Contributions

### Bug Fixes

- Check existing issues for related bug reports
- Create a new issue if none exists
- Include tests that reproduce the bug
- Ensure all tests pass after the fix

### New Features

- Discuss large changes in an issue first
- Follow the existing architecture
- Include comprehensive tests
- Update documentation

### Documentation

- Fix typos and grammatical errors
- Improve clarity of existing documentation
- Add examples and tutorials
- Translate documentation (if applicable)

## Submitting Pull Requests

### Before Submitting

1. Ensure your code follows the style guidelines
2. Run all tests and ensure they pass
3. Update relevant documentation
4. Rebase your branch on the latest main branch

### Pull Request Template

Use this template for your pull request description:

```markdown
## Description
Brief description of the changes

## Type of Change
- [ ] Bug fix
- [ ] New feature
- [ ] Breaking change
- [ ] Documentation update

## Testing
- [ ] Tests pass locally
- [ ] Added new tests for new functionality
- [ ] Updated documentation

## Checklist
- [ ] Code follows project style guidelines
- [ ] Self-review completed
- [ ] Documentation updated
```

## Getting Help

- Create an issue for questions or problems
- Join our discussions (if available)
- Check existing issues and documentation

## License

By contributing to this project, you agree that your contributions will be licensed under the same license as the project (MIT License).
