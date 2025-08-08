# 🔐 Secure Thermal Image AI Analyzer

This version of the Thermal Image AI Analyzer has been updated for secure deployment with proper token management.

## 🚀 Quick Start (Secure)

### 1. Automatic Setup
```bash
python setup.py
```
This will guide you through the setup process and create your `.env` file securely.

### 2. Manual Setup
1. Copy the environment template:
   ```bash
   cp env_template.txt .env
   ```

2. Edit `.env` and add your Hugging Face token:
   ```
   HUGGINGFACE_TOKEN=hf_your_actual_token_here
   ```

3. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

## 🔑 Getting Your Hugging Face Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Click "New token"
3. Give it a name (e.g., "Thermal Analyzer")
4. Select "Read" permissions
5. Copy the token (starts with `hf_`)

## 🛡️ Security Features

- ✅ **No hardcoded tokens** - All sensitive data moved to environment variables
- ✅ **Environment file protection** - `.env` is in `.gitignore`
- ✅ **Configuration validation** - App checks for required environment variables
- ✅ **Secure deployment ready** - Works with Streamlit Cloud, Heroku, Docker, etc.

## 📁 File Structure

```
Thermal/
├── streamlit_app.py          # Main application (no tokens)
├── config.py                 # Configuration management
├── .env                      # Environment variables (create this)
├── env_template.txt          # Template for .env
├── .gitignore               # Protects sensitive files
├── setup.py                 # Automated setup script
├── DEPLOYMENT.md            # Detailed deployment guide
└── requirements.txt         # Dependencies
```

## 🌐 Deployment Options

### Streamlit Cloud (Recommended)
1. Push to GitHub (`.env` is automatically ignored)
2. Connect to [Streamlit Cloud](https://share.streamlit.io/)
3. Add `HUGGINGFACE_TOKEN` in app settings
4. Deploy!

### Local Development
```bash
python setup.py
streamlit run streamlit_app.py
```

### Docker
```bash
docker build -t thermal-analyzer .
docker run -p 8501:8501 -e HUGGINGFACE_TOKEN=your_token thermal-analyzer
```

## 🔧 Configuration

All settings are now in environment variables:

| Variable | Description | Default |
|----------|-------------|---------|
| `HUGGINGFACE_TOKEN` | Your HF token | Required |
| `USE_GPU` | Enable GPU | `true` |
| `LOW_MEMORY_MODE` | Low memory mode | `false` |
| `DEFAULT_MODEL` | Default AI model | `BLIP Base` |

## 🐛 Troubleshooting

### "HUGGINGFACE_TOKEN environment variable is required"
- Run `python setup.py` to configure your token
- Or manually create `.env` file with your token

### "Model loading failed"
- Check your token has read permissions
- Verify internet connection
- Try a different model

### "Out of memory"
- Set `LOW_MEMORY_MODE=true` in `.env`
- Use `DEFAULT_MODEL=BLIP Base`

## 📚 Documentation

- [DEPLOYMENT.md](DEPLOYMENT.md) - Complete deployment guide
- [Original README.md](README.md) - Original documentation

## 🔄 Migration from Old Version

If you have the old version with hardcoded tokens:

1. **Backup your token** from the old code
2. **Delete the old token** from `streamlit_app.py`
3. **Run setup**: `python setup.py`
4. **Add your token** when prompted

## 🛡️ Security Best Practices

1. **Never commit tokens** - Always use environment variables
2. **Use .gitignore** - Ensures `.env` is never committed
3. **Rotate tokens** - Regularly update your Hugging Face token
4. **Limit permissions** - Use read-only tokens when possible
5. **Monitor usage** - Check your token usage regularly

## 📞 Support

If you encounter issues:

1. Check the [DEPLOYMENT.md](DEPLOYMENT.md) troubleshooting section
2. Verify your environment variables are set correctly
3. Test with a simple model first (BLIP Base)
4. Check your Hugging Face token permissions

---

**Note**: This version is production-ready and secure for deployment. The original hardcoded token has been removed and replaced with proper environment variable management.
