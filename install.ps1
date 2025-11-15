# SIE-X Installation Script for Windows
# PowerShell version

Write-Host "🚀 Installing SIE-X..." -ForegroundColor Green

# Create virtual environment
Write-Host "`n📦 Creating Python virtual environment..." -ForegroundColor Cyan
python -m venv venv

# Activate virtual environment
Write-Host "🔧 Activating virtual environment..." -ForegroundColor Cyan
& ".\venv\Scripts\Activate.ps1"

# Install Python dependencies
Write-Host "`n📚 Installing Python dependencies..." -ForegroundColor Cyan
python -m pip install --upgrade pip
pip install -r requirements.txt

# Download spaCy models
Write-Host "`n🧠 Downloading spaCy models..." -ForegroundColor Cyan
python -m spacy download en_core_web_lg
python -m spacy download xx_ent_wiki_sm

# Setup Node.js SDK
Write-Host "`n📦 Setting up Node.js SDK..." -ForegroundColor Cyan
Set-Location sdk\nodejs
npm install
npm run build
Set-Location ..\..

# Setup Go SDK
Write-Host "`n📦 Setting up Go SDK..." -ForegroundColor Cyan
Set-Location sdk\go
go mod download
Set-Location ..\..

# Setup Admin Dashboard
Write-Host "`n📦 Setting up Admin Dashboard..." -ForegroundColor Cyan
Set-Location admin-dashboard
npm install
Set-Location ..

# Create necessary directories
Write-Host "`n📁 Creating necessary directories..." -ForegroundColor Cyan
$directories = @("models", "data", "logs", "outputs")
foreach ($dir in $directories)
{
    if (!(Test-Path $dir))
    {
        New-Item -ItemType Directory -Path $dir | Out-Null
        Write-Host "  ✓ Created $dir" -ForegroundColor Gray
    }
    else
    {
        Write-Host "  ✓ $dir already exists" -ForegroundColor Gray
    }
}

Write-Host "`n✅ Installation complete!" -ForegroundColor Green
Write-Host "📖 Run 'uvicorn sie_x.api.server:app --reload' to start the API" -ForegroundColor Yellow
Write-Host "💡 Don't forget to activate the virtual environment: .\venv\Scripts\Activate.ps1" -ForegroundColor Yellow

