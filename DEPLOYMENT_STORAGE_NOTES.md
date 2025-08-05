# Deployment Storage Considerations

## Current Limitation

The current implementation saves recipes to a local `/recipes` directory, which **will not work** in Streamlit Community Cloud because:

1. **Ephemeral Storage**: Container storage is temporary and resets on restart
2. **Multi-user Conflicts**: All users would share the same filesystem
3. **No Persistence**: Recipes are lost when the container restarts

## Temporary Workaround

For initial deployment, the app will work but with these limitations:
- Recipes are saved temporarily during the session
- Users should download recipes immediately after extraction
- Recipes won't persist between sessions
- The Recipe Viewer page won't show historical recipes

## Recommended Solutions

### 1. Session-Only Mode (Quickest)
Modify the app to:
- Store recipes in `st.session_state`
- Remove filesystem saves
- Provide download buttons for JSON/TXT/ZIP
- Show recipes only for current session

### 2. Cloud Storage Integration
Add support for:
- AWS S3 / Google Cloud Storage / Azure Blob
- User authentication to separate data
- Persistent recipe storage
- Image hosting

### 3. Database Backend
Implement:
- PostgreSQL/MySQL for recipe metadata
- Object storage for images
- User accounts and authentication

## Environment Variables

Make sure to add these to Streamlit Secrets (not .env):
```toml
OPENAI_API_KEY = "your-key-here"
OPENAI_MODEL = "o4-mini"  # optional

# If implementing cloud storage:
# AWS_ACCESS_KEY_ID = "..."
# AWS_SECRET_ACCESS_KEY = "..."
# S3_BUCKET_NAME = "..."
```

## Next Steps

1. Deploy current version with download-focused workflow
2. Plan storage backend based on usage needs
3. Implement persistent storage in future update