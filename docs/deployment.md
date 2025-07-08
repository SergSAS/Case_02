# Deployment Guide

This guide covers various deployment scenarios for the Text Classification Analysis system.

## Table of Contents

1. [Local Development](#local-development)
2. [Production Deployment](#production-deployment)
3. [Docker Deployment](#docker-deployment)
4. [Cloud Deployment](#cloud-deployment)
5. [CI/CD Pipeline](#cicd-pipeline)
6. [Monitoring](#monitoring)

## Local Development

### Prerequisites

- Python 3.8 or higher
- pip package manager
- Git
- 8GB RAM minimum
- 5GB free disk space

### Step-by-Step Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/yourusername/text-classification-analysis.git
   cd text-classification-analysis
   ```

2. **Create virtual environment:**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate

   # Linux/Mac
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment:**
   ```bash
   cp config/.env.example config/.env
   ```

5. **Edit configuration:**
   ```bash
   # Edit config/.env with your API keys
   # Required: GROQ_API_KEY for LLM functionality
   ```

6. **Verify installation:**
   ```bash
   python main.py --mode traditional --force-reload
   ```

### Development Tools Setup

1. **Install development dependencies:**
   ```bash
   pip install -r requirements-dev.txt
   ```

2. **Setup pre-commit hooks:**
   ```bash
   pre-commit install
   ```

3. **Run code quality checks:**
   ```bash
   black src/
   flake8 src/
   mypy src/
   ```

## Production Deployment

### System Requirements

- **CPU:** 4+ cores recommended
- **RAM:** 16GB minimum
- **Storage:** 20GB SSD
- **OS:** Ubuntu 20.04+ or similar

### Production Setup

1. **Create dedicated user:**
   ```bash
   sudo adduser textclassify
   sudo usermod -aG sudo textclassify
   su - textclassify
   ```

2. **Install system dependencies:**
   ```bash
   sudo apt update
   sudo apt install python3.9 python3.9-venv python3-pip nginx supervisor
   ```

3. **Clone and setup:**
   ```bash
   cd /opt
   sudo git clone https://github.com/yourusername/text-classification-analysis.git
   sudo chown -R textclassify:textclassify text-classification-analysis
   cd text-classification-analysis
   ```

4. **Create production virtual environment:**
   ```bash
   python3.9 -m venv venv
   source venv/bin/activate
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

5. **Configure for production:**
   ```bash
   # Create production config
   cp config/.env.example config/.env.production
   
   # Edit with production values
   nano config/.env.production
   ```

### Running as a Service

1. **Create systemd service file:**
   ```bash
   sudo nano /etc/systemd/system/textclassify.service
   ```

2. **Add service configuration:**
   ```ini
   [Unit]
   Description=Text Classification Analysis Service
   After=network.target

   [Service]
   Type=simple
   User=textclassify
   WorkingDirectory=/opt/text-classification-analysis
   Environment="PATH=/opt/text-classification-analysis/venv/bin"
   ExecStart=/opt/text-classification-analysis/venv/bin/python main.py --mode compare
   Restart=always

   [Install]
   WantedBy=multi-user.target
   ```

3. **Enable and start service:**
   ```bash
   sudo systemctl daemon-reload
   sudo systemctl enable textclassify
   sudo systemctl start textclassify
   ```

## Docker Deployment

### Dockerfile

Create `Dockerfile` in project root:

```dockerfile
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements first for better caching
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY . .

# Create non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser:appuser /app
USER appuser

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Expose port if needed (for future API)
EXPOSE 8000

# Default command
CMD ["python", "main.py", "--mode", "full"]
```

### Docker Compose

Create `docker-compose.yml`:

```yaml
version: '3.8'

services:
  textclassify:
    build: .
    container_name: text-classification
    environment:
      - GROQ_API_KEY=${GROQ_API_KEY}
      - LLM_MODEL=gemma2-9b-it
    volumes:
      - ./config:/app/config
      - ./data:/app/data
      - ./results:/app/results
      - ./logs:/app/logs
    restart: unless-stopped
    command: python main.py --mode full --visualize
```

### Running with Docker

1. **Build image:**
   ```bash
   docker build -t text-classification:latest .
   ```

2. **Run container:**
   ```bash
   docker run -d \
     --name text-classification \
     -v $(pwd)/config:/app/config \
     -v $(pwd)/results:/app/results \
     -e GROQ_API_KEY=$GROQ_API_KEY \
     text-classification:latest
   ```

3. **Using docker-compose:**
   ```bash
   docker-compose up -d
   ```

## Cloud Deployment

### AWS EC2 Deployment

1. **Launch EC2 instance:**
   - AMI: Ubuntu Server 20.04 LTS
   - Instance type: t3.large (minimum)
   - Storage: 30GB GP3

2. **Security group rules:**
   - SSH (22) from your IP
   - HTTP (80) if adding web interface
   - HTTPS (443) if adding web interface

3. **Connect and deploy:**
   ```bash
   ssh -i your-key.pem ubuntu@your-ec2-ip
   
   # Follow production deployment steps
   ```

### Google Cloud Platform

1. **Create Compute Engine instance:**
   ```bash
   gcloud compute instances create text-classification \
     --machine-type=n1-standard-4 \
     --image-family=ubuntu-2004-lts \
     --image-project=ubuntu-os-cloud \
     --boot-disk-size=30GB
   ```

2. **Deploy application:**
   ```bash
   gcloud compute ssh text-classification
   
   # Follow production deployment steps
   ```

### Azure Virtual Machine

1. **Create VM:**
   ```bash
   az vm create \
     --resource-group myResourceGroup \
     --name text-classification \
     --image UbuntuLTS \
     --size Standard_D4s_v3 \
     --admin-username azureuser \
     --generate-ssh-keys
   ```

2. **Deploy:**
   ```bash
   ssh azureuser@vm-public-ip
   
   # Follow production deployment steps
   ```

## CI/CD Pipeline

### GitHub Actions

Create `.github/workflows/deploy.yml`:

```yaml
name: Deploy

on:
  push:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
    - uses: actions/checkout@v3
    
    - name: Set up Python
      uses: actions/setup-python@v4
      with:
        python-version: '3.9'
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov
    
    - name: Run tests
      run: pytest tests/
    
    - name: Check code quality
      run: |
        pip install black flake8
        black --check src/
        flake8 src/

  deploy:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    
    steps:
    - uses: actions/checkout@v3
    
    - name: Deploy to server
      uses: appleboy/ssh-action@master
      with:
        host: ${{ secrets.HOST }}
        username: ${{ secrets.USERNAME }}
        key: ${{ secrets.SSH_KEY }}
        script: |
          cd /opt/text-classification-analysis
          git pull origin main
          source venv/bin/activate
          pip install -r requirements.txt
          sudo systemctl restart textclassify
```

## Monitoring

### Application Monitoring

1. **Setup logging aggregation:**
   ```python
   # In config/.env
   LOG_LEVEL=INFO
   LOG_FILE=/var/log/textclassify/app.log
   ```

2. **Log rotation:**
   ```bash
   sudo nano /etc/logrotate.d/textclassify
   ```

   ```
   /var/log/textclassify/*.log {
       daily
       rotate 14
       compress
       delaycompress
       notifempty
       create 0640 textclassify textclassify
       sharedscripts
   }
   ```

3. **Health check endpoint (future):**
   ```python
   # Add to main.py
   @app.route('/health')
   def health_check():
       return {'status': 'healthy', 'timestamp': datetime.now()}
   ```

### Performance Monitoring

1. **System metrics:**
   ```bash
   # Install monitoring tools
   sudo apt install htop iotop nethogs
   ```

2. **Application metrics:**
   - Track processing times
   - Monitor API usage
   - Log memory consumption

3. **Alerts setup:**
   - Configure email alerts for errors
   - Set up Slack notifications
   - Monitor API rate limits

## Troubleshooting

### Common Issues

1. **Memory errors:**
   ```bash
   # Increase swap
   sudo fallocate -l 4G /swapfile
   sudo chmod 600 /swapfile
   sudo mkswap /swapfile
   sudo swapon /swapfile
   ```

2. **Permission errors:**
   ```bash
   # Fix permissions
   sudo chown -R textclassify:textclassify /opt/text-classification-analysis
   ```

3. **API rate limits:**
   ```python
   # Adjust in config/.env
   RATE_LIMIT_DELAY=5.0
   ```

### Backup Strategy

1. **Backup script:**
   ```bash
   #!/bin/bash
   BACKUP_DIR="/backup/textclassify"
   DATE=$(date +%Y%m%d_%H%M%S)
   
   # Backup data and results
   tar -czf $BACKUP_DIR/data_$DATE.tar.gz /opt/text-classification-analysis/data
   tar -czf $BACKUP_DIR/results_$DATE.tar.gz /opt/text-classification-analysis/results
   
   # Keep only last 7 days
   find $BACKUP_DIR -name "*.tar.gz" -mtime +7 -delete
   ```

2. **Cron job:**
   ```bash
   0 2 * * * /opt/scripts/backup-textclassify.sh
   ``` 