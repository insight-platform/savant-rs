# Savant Info

A simple utility to retrieve and output Savant-rs package information for use in scripts and build processes.

## Usage

### Basic Usage

Run without arguments to display all available information:

```bash
cargo run -p savant_info
```

This will output all information with shell-compatible formatting:

```
SAVANT_RS_NAME='savant_info'
SAVANT_RS_VERSION='1.0.2'
SAVANT_RS_LICENSE='Apache-2.0'
SAVANT_RS_AUTHORS='Ivan Kudriavtsev <ivan.a.kudryavtsev@gmail.com>'
SAVANT_RS_DESCRIPTION='Savant Rust core functions library'
SAVANT_RS_HOMEPAGE='https://github.com/insight-platform/savant-rs'
SAVANT_RS_REPOSITORY='https://github.com/insight-platform/savant-rs'
SAVANT_RS_RUST_VERSION='1.83'
```

### Filtering Information

Specify one or more parameters to only retrieve those specific values:

```bash
cargo run -p savant_info version repository
```

Output:
```
SAVANT_RS_VERSION='1.0.2'
SAVANT_RS_REPOSITORY='https://github.com/insight-platform/savant-rs'
```

### Available Parameters

The following parameters can be specified:

- `name` - Package name
- `version` - Package version
- `license` - Package license
- `authors` - Package authors
- `description` - Package description
- `homepage` - Package homepage URL
- `repository` - Package repository URL
- `keywords` - Package keywords
- `rust_version` - Required Rust version

### Using in Shell Scripts

The output is formatted to be directly usable in shell scripts:

```bash
# Load all variables
eval $(cargo run -p savant_info)

# Use the variables
echo "Running Savant version: $SAVANT_RS_VERSION"
echo "Repository: $SAVANT_RS_REPOSITORY"

# Alternatively, load only specific variables
eval $(cargo run -p savant_info version license)
echo "Savant version $SAVANT_RS_VERSION is licensed under $SAVANT_RS_LICENSE"
```

### Integration in Build Scripts

This utility is particularly useful for build scripts and CI/CD pipelines where you need
to access package metadata:

```bash
#!/bin/bash
# Example CI script

# Get version information
eval $(cargo run -p savant_info version)

# Use in Docker build
docker build -t savant:$SAVANT_RS_VERSION .
```

## Notes

- All values are properly escaped for shell compatibility
- All environment variable names are prefixed with `SAVANT_RS_` and capitalized
- Unknown parameters will produce error messages on stderr 