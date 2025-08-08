# 🔐 Token Setup Guide

This guide explains the new token management system for the Thermal Image AI Analyzer.

## 🎯 Problem Solved

Previously, tokens were stored in `.env` files which could accidentally be committed to GitHub. The new system separates sensitive tokens into a dedicated `secrets.py` file that is automatically ignored by git.

## 📁 New File Structure

```
Thermal/
├── secrets_template.py    # Template file (safe to commit)
├── secrets.py            # Actual secrets (gitignored)
├── setup_tokens.py       # Interactive setup script
├── test_config.py        # Configuration test script
├── config.py             # Updated configuration
└── .gitignore           # Updated to ignore secrets.py
```

## 🔧 How It Works

### 1. Configuration Priority
The system tries to load tokens in this order:
1. `secrets.py` file (preferred)
2. Environment variables (fallback)
3. `.env` file (legacy support)

### 2. Security Features
- `secrets.py` is automatically ignored by git
- `secrets_template.py` can be safely committed (contains only placeholders)
- Interactive setup script prevents accidental token exposure

## 🚀 Setup Options

### Option A: Interactive Setup (Recommended)
```bash
python setup_tokens.py
```
This script will:
- Guide you through getting a Hugging Face token
- Create `secrets.py` with your token
- Test the configuration
- Provide clear instructions

### Option B: Manual Setup
1. Copy the template:
   ```bash
   cp secrets_template.py secrets.py
   ```
2. Edit `secrets.py` and replace `"your_actual_huggingface_token_here"` with your real token
3. Test the configuration:
   ```bash
   python test_config.py
   ```

## 🔍 Testing Your Setup

Run the test script to verify everything is working:
```bash
python test_config.py
```

This will check:
- ✅ Token file existence
- ✅ Configuration import
- ✅ Token validation
- ✅ Model configuration
- ✅ Device settings

## 🔒 Security Best Practices

1. **Never commit `secrets.py`** - It's automatically ignored by git
2. **Use different tokens** for development and production
3. **Rotate tokens regularly** for security
4. **Keep tokens private** - don't share them in issues or discussions

## 🆘 Troubleshooting

### "Token validation failed"
- Run `python setup_tokens.py` to configure your token
- Make sure you have a valid Hugging Face token
- Check that `secrets.py` contains your actual token, not the placeholder

### "Config import failed"
- Make sure all dependencies are installed: `pip install -r requirements.txt`
- Check that `config.py` is in the same directory

### "secrets.py not found"
- This is normal if you haven't set up tokens yet
- The system will fall back to environment variables
- Run `python setup_tokens.py` to create the file

## 📋 Migration from .env

If you were previously using a `.env` file:

1. **Keep your `.env` file** - it will still work as a fallback
2. **Run the setup script** to create `secrets.py`:
   ```bash
   python setup_tokens.py
   ```
3. **Test the configuration**:
   ```bash
   python test_config.py
   ```
4. **Optionally remove `.env`** once `secrets.py` is working

## 🔄 Backward Compatibility

The new system maintains full backward compatibility:
- `.env` files still work
- Environment variables still work
- Existing configurations continue to function
- No breaking changes to the application

## 📞 Support

If you encounter issues:
1. Run `python test_config.py` to diagnose problems
2. Check the error messages for specific guidance
3. Ensure your Hugging Face token is valid and has the correct permissions
4. Create an issue on GitHub if problems persist
