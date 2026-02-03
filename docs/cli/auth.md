# Authentication

The `mlxsmith auth` commands manage your Hugging Face token for pulling models and datasets.

## Login

Provide a token interactively or via `--token` / `HF_TOKEN`:

```bash
mlxsmith auth login
```

```bash
mlxsmith auth login --token $HF_TOKEN
```

Add `--validate` to verify the token with the Hugging Face API (enabled by default for login):

```bash
mlxsmith auth login --validate
```

## Status

Check whether a token is stored, and optionally validate it:

```bash
mlxsmith auth status
mlxsmith auth status --validate
```

## Logout

Remove the stored token:

```bash
mlxsmith auth logout
```

Tokens are stored in the standard Hugging Face location (respecting `HF_HOME` when set).
