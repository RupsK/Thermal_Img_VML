# 🚀 Deployment Guide for Thermal Image AI Analyzer

This guide will help you deploy the Thermal Image AI Analyzer securely without exposing sensitive tokens.

## 🔐 Security Setup

### 1. Get Your Hugging Face Token

1. Go to [Hugging Face Settings](https://huggingface.co/settings/tokens)
2. Create a new token with "Read" permissions
3. Copy the token (it starts with `hf_`)

### 2. Create Environment File

1. Copy `env_template.txt` to `.env`:
   ```bash
   cp env_template.txt .env
   ```

2. Edit `.env` and replace `your_huggingface_token_here` with your actual token:
   ```
   HUGGINGFACE_TOKEN=hf_your_actual_token_here
   ```

3. **IMPORTANT**: Add `.env` to your `.gitignore` file to prevent it from being committed:
   ```bash
   echo ".env" >> .gitignore
   ```

## 🌐 Deployment Options

### Option 1: Local Deployment

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run the app:
   ```bash
   streamlit run streamlit_app.py
   ```

3. Or use the batch file (Windows):
   ```bash
   start_app.bat
   ```

### Option 2: Streamlit Cloud Deployment

1. Push your code to GitHub (make sure `.env` is in `.gitignore`)

2. Go to [Streamlit Cloud](https://share.streamlit.io/)

3. Connect your GitHub repository

4. Set environment variables in Streamlit Cloud:
   - Go to your app settings
   - Add `HUGGINGFACE_TOKEN` with your token value

5. Deploy!

### Option 3: Docker Deployment

1. Create a `Dockerfile`:
   ```dockerfile
   FROM python:3.9-slim
   
   WORKDIR /app
   
   COPY requirements.txt .
   RUN pip install -r requirements.txt
   
   COPY . .
   
   EXPOSE 8501
   
   CMD ["streamlit", "run", "streamlit_app.py", "--server.port=8501", "--server.address=0.0.0.0"]
   ```

2. Build and run:
   ```bash
   docker build -t thermal-analyzer .
   docker run -p 8501:8501 -e HUGGINGFACE_TOKEN=your_token thermal-analyzer
   ```

### Option 4: Heroku Deployment

1. Create `Procfile`:
   ```
   web: streamlit run streamlit_app.py --server.port=$PORT --server.address=0.0.0.0
   ```

2. Create `setup.sh`:
   ```bash
   mkdir -p ~/.streamlit/
   echo "\
   [server]\n\
   headless = true\n\
   port = $PORT\n\
   enableCORS = false\n\
   \n\
   " > ~/.streamlit/config.toml
   ```

3. Deploy to Heroku and set environment variables:
   ```bash
   heroku config:set HUGGINGFACE_TOKEN=your_token
   ```

## 🔧 Configuration Options

### Environment Variables

| Variable | Description | Default |
|----------|-------------|---------|
| `HUGGINGFACE_TOKEN` | Your Hugging Face API token | Required |
| `USE_GPU` | Enable GPU acceleration | `true` |
| `LOW_MEMORY_MODE` | Enable low memory mode | `false` |
| `DEFAULT_MODEL` | Default AI model to use | `BLIP Base` |

### Model Options

- **BLIP Base**: Fast, good for general analysis
- **BLIP Large**: More accurate, slower
- **GIT Base**: Good for detailed descriptions
- **LLaVA-Next**: Advanced vision-language model
- **SmolVLM**: Lightweight, fast processing

## 🛡️ Security Best Practices

1. **Never commit tokens**: Always use environment variables
2. **Use .gitignore**: Ensure `.env` is ignored
3. **Rotate tokens**: Regularly update your Hugging Face token
4. **Limit permissions**: Use read-only tokens when possible
5. **Monitor usage**: Check your Hugging Face token usage

## 📊 Performance Optimization

### For Low Memory Systems

Set in `.env`:
```
LOW_MEMORY_MODE=true
USE_GPU=false
DEFAULT_MODEL=BLIP Base
```

### For High Performance Systems

Set in `.env`:
```
LOW_MEMORY_MODE=false
USE_GPU=true
DEFAULT_MODEL=LLaVA-Next
```

## 🐛 Troubleshooting

### Common Issues

1. **"HUGGINGFACE_TOKEN environment variable is required"**
   - Check that your `.env` file exists and contains the token
   - Verify the token format starts with `hf_`

2. **"Model loading failed"**
   - Check your internet connection
   - Verify your Hugging Face token has read permissions
   - Try a different model in the configuration

3. **"Out of memory"**
   - Enable `LOW_MEMORY_MODE=true`
   - Use a smaller model like "BLIP Base"
   - Disable GPU with `USE_GPU=false`

### Getting Help

1. Check the logs for detailed error messages
2. Verify your environment variables are set correctly
3. Test with a simple model first (BLIP Base)
4. Check your Hugging Face token permissions

## 📝 License

This project is open source. Please ensure you comply with the licenses of the AI models you use.
