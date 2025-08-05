# Streamlit Community Cloud Deployment Guide

This guide explains how to deploy the AI Recipes app to Streamlit Community Cloud using uv for dependency management.

## Prerequisites

1. A GitHub account
2. Your repository pushed to GitHub
3. An OpenAI API key

## Dependency Management with uv

This project uses `uv` for dependency management. Dependencies are defined in `pyproject.toml` and locked in `uv.lock`. For deployment compatibility, a `requirements.txt` file is automatically generated from these files.

### Automatic requirements.txt Generation

A GitHub Action automatically generates `requirements.txt` whenever:
- Changes are pushed to `pyproject.toml` or `uv.lock`
- The workflow is manually triggered

The generated `requirements.txt` excludes development dependencies to keep the deployment lean.

### Manual Generation (if needed)

To manually generate requirements.txt locally:

```bash
# Export production dependencies only
uv export --no-dev --format requirements-txt -o requirements.txt
```

## Deployment Steps

1. **Ensure requirements.txt is up to date**:
   - Check that the GitHub Action has run and `requirements.txt` exists
   - Or generate it manually and commit it

2. **Go to Streamlit Community Cloud**:
   - Visit [share.streamlit.io](https://share.streamlit.io)
   - Sign in with your GitHub account

3. **Deploy your app**:
   - Click "New app"
   - Select your repository: `cadenzadesigns/ai-recipes`
   - Branch: `main` (or your current branch)
   - Main file path: `streamlit_app.py`
   - Click "Deploy"

4. **Configure secrets**:
   - Once deployed, go to your app settings (⋮ menu → Settings)
   - Navigate to the "Secrets" section
   - Add your secrets in TOML format:
   ```toml
   OPENAI_API_KEY = "your-openai-api-key-here"
   
   # Optional: Configure OpenAI model
   OPENAI_MODEL = "o4-mini"
   
   # Optional: Paprika credentials
   # PAPRIKA_EMAIL = "your-email@example.com"
   # PAPRIKA_PASSWORD = "your-password"
   ```
   - Save the secrets

5. **Custom subdomain** (optional):
   - In app settings, you can set a custom subdomain
   - For example: `ai-recipes.streamlit.app`

## File Structure for Deployment

The following files are required for Streamlit deployment:

- `streamlit_app.py` - Main entry point
- `requirements.txt` - Python dependencies (auto-generated from uv)
- `packages.txt` - System dependencies (apt packages)
- `.streamlit/config.toml` - Streamlit configuration
- Your app code in the `app/` and `src/` directories

## Adding New Dependencies

When you need to add new dependencies:

1. Add them using uv:
   ```bash
   uv add package-name
   ```

2. Commit the updated `pyproject.toml` and `uv.lock`

3. The GitHub Action will automatically update `requirements.txt`

4. Streamlit Community Cloud will automatically redeploy with the new dependencies

## Troubleshooting

### ModuleNotFoundError
If you get import errors, ensure:
- The GitHub Action has generated an up-to-date `requirements.txt`
- All dependencies in `pyproject.toml` are included (check that `--no-dev` isn't excluding needed packages)

### OpenCV Issues
The deployment already uses `opencv-python-headless` to avoid GUI dependencies.

### File Upload Limits
The config sets a 200MB upload limit. You can adjust this in `.streamlit/config.toml`.

### Secrets Not Working
Ensure secrets are properly formatted as TOML and saved in the app settings.

### HEIC Support Issues
If HEIC image support doesn't work in deployment, ensure the `pillow-heif` extra is installed:
- Check that `ai-recipes[heic]` is in the generated requirements.txt
- Or explicitly add `pillow-heif` to dependencies

## Local Testing

To test the deployment setup locally:

```bash
# Install dependencies using uv
uv sync

# Run the app
uv run streamlit run streamlit_app.py
```

## Important Limitations

### Recipe Storage
**WARNING**: Recipes saved in Streamlit Community Cloud are **temporary**:
- The `/recipes` directory is not persistent
- Recipes will be lost when the app restarts
- All users share the same temporary filesystem

**Workaround**: Always download recipes immediately after extraction using the download buttons.

### Environment Variables
The `.env` file is NOT deployed. You must add secrets manually in Streamlit Community Cloud settings as shown in step 4 above.

## Notes

- The app will automatically reload when you push changes to GitHub
- Logs are available in the app menu (⋮ → Logs)
- The free tier has resource limits but should be sufficient for personal use
- The GitHub Action ensures `requirements.txt` stays in sync with your uv-managed dependencies
- For persistent storage, consider integrating cloud storage (S3, GCS) or a database in a future update