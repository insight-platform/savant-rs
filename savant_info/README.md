# savant-info

CLI helper that prints [Savant](https://github.com/insight-platform/savant-rs) build and package metadata (name, version, license, authors, homepage, repository, rust-version, ...) in a shell-quotable form. Designed for CI scripts, release tooling, and Dockerfile stamping.

## Install

```sh
cargo install savant-info
```

## Usage

Print everything:

```sh
savant-info
```

```sh
SAVANT_RS_NAME='savant-info'
SAVANT_RS_VERSION='2.1.0'
SAVANT_RS_LICENSE='Apache-2.0'
SAVANT_RS_AUTHORS='Ivan Kudriavtsev <ivan.a.kudryavtsev@gmail.com>'
SAVANT_RS_HOMEPAGE='https://insight-platform.github.io/savant-rs/'
SAVANT_RS_REPOSITORY='https://github.com/insight-platform/savant-rs'
SAVANT_RS_RUST_VERSION='1.83'
```

Print only specific fields:

```sh
savant-info version repository
```

```sh
SAVANT_RS_VERSION='2.1.0'
SAVANT_RS_REPOSITORY='https://github.com/insight-platform/savant-rs'
```

Use in shell scripts:

```sh
eval "$(savant-info version license)"
echo "Savant ${SAVANT_RS_VERSION} (${SAVANT_RS_LICENSE})"
```

Use in a Dockerfile stamp step:

```sh
eval "$(savant-info version)"
docker build -t myorg/savant-app:"${SAVANT_RS_VERSION}" .
```

Supported fields: `name`, `version`, `license`, `authors`, `description`, `homepage`, `repository`, `keywords`, `rust_version`.

All values are safely single-quoted for direct shell consumption. Unknown field names are reported on stderr and do not abort the run.

## Documentation

- [Savant project site](https://insight-platform.github.io/savant-rs/)
- [Source](https://github.com/insight-platform/savant-rs/tree/main/savant_info)

## License

Licensed under the [Apache License, Version 2.0](https://github.com/insight-platform/savant-rs/blob/main/LICENSE).
