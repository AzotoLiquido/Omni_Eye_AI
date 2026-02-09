# Omni Eye AI - Script di Avvio PowerShell
# Avvio avanzato con controlli automatici

param(
    [switch]$SkipChecks,
    [switch]$InstallDeps,
    [string]$Model = "llama3.2"
)

$ErrorActionPreference = "Stop"

function Write-Header {
    Write-Host "`n" -NoNewline
    Write-Host "====================================================================" -ForegroundColor Cyan
    Write-Host "        🤖 OMNI EYE AI - Assistente AI Locale" -ForegroundColor Green
    Write-Host "====================================================================" -ForegroundColor Cyan
    Write-Host "    Completamente privato • Gratuito • Illimitato" -ForegroundColor Gray
    Write-Host "====================================================================" -ForegroundColor Cyan
    Write-Host ""
}

function Test-OllamaInstalled {
    Write-Host "🔍 Verifica Ollama..." -NoNewline
    try {
        $null = Get-Command ollama -ErrorAction Stop
        Write-Host " ✅" -ForegroundColor Green
        return $true
    } catch {
        Write-Host " ❌" -ForegroundColor Red
        Write-Host "   Ollama non trovato!" -ForegroundColor Yellow
        Write-Host "   Installa con: winget install Ollama.Ollama" -ForegroundColor Yellow
        return $false
    }
}

function Test-OllamaRunning {
    Write-Host "🔍 Verifica servizio Ollama..." -NoNewline
    try {
        $result = ollama list 2>&1
        Write-Host " ✅" -ForegroundColor Green
        return $true
    } catch {
        Write-Host " ❌" -ForegroundColor Red
        return $false
    }
}

function Get-InstalledModels {
    try {
        $output = ollama list 2>&1 | Out-String
        $lines = $output -split "`n" | Select-Object -Skip 1
        $models = @()
        
        foreach ($line in $lines) {
            if ($line.Trim()) {
                $modelName = ($line -split '\s+')[0]
                if ($modelName) {
                    $models += $modelName
                }
            }
        }
        
        return $models
    } catch {
        return @()
    }
}

function Install-Model {
    param([string]$ModelName)
    
    Write-Host "`n📦 Download modello '$ModelName'..." -ForegroundColor Cyan
    Write-Host "   (Questo può richiedere alcuni minuti)`n" -ForegroundColor Gray
    
    try {
        ollama pull $ModelName
        Write-Host "`n✅ Modello installato!" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "`n❌ Errore nel download" -ForegroundColor Red
        return $false
    }
}

function Test-PythonDependencies {
    Write-Host "🔍 Verifica dipendenze Python..." -NoNewline
    
    $required = @('flask', 'ollama', 'flask_cors')
    $missing = @()
    
    foreach ($package in $required) {
        try {
            python -c "import $package" 2>$null
            if ($LASTEXITCODE -ne 0) {
                $missing += $package
            }
        } catch {
            $missing += $package
        }
    }
    
    if ($missing.Count -eq 0) {
        Write-Host " ✅" -ForegroundColor Green
        return $true
    } else {
        Write-Host " ⚠️" -ForegroundColor Yellow
        Write-Host "   Mancanti: $($missing -join ', ')" -ForegroundColor Yellow
        return $false
    }
}

function Install-Dependencies {
    Write-Host "`n📦 Installazione dipendenze..." -ForegroundColor Cyan
    try {
        python -m pip install -r requirements.txt --quiet
        Write-Host "✅ Dipendenze installate!" -ForegroundColor Green
        return $true
    } catch {
        Write-Host "❌ Errore installazione" -ForegroundColor Red
        return $false
    }
}

# Main Script
Clear-Host
Write-Header

# Cambio directory
Set-Location $PSScriptRoot

# Controlli preliminari
if (-not $SkipChecks) {
    
    # Test Ollama
    $ollamaInstalled = Test-OllamaInstalled
    if (-not $ollamaInstalled) {
        Write-Host "`n❌ Installa Ollama prima di continuare" -ForegroundColor Red
        Write-Host "   Download: https://ollama.ai/download`n" -ForegroundColor Cyan
        pause
        exit 1
    }
    
    $ollamaRunning = Test-OllamaRunning
    if (-not $ollamaRunning) {
        Write-Host "⚠️  Ollama non risponde, provo ad avviarlo..." -ForegroundColor Yellow
        Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
        Start-Sleep -Seconds 3
    }
    
    # Test modelli
    Write-Host "📦 Modelli installati:" -ForegroundColor Cyan
    $models = Get-InstalledModels
    
    if ($models.Count -eq 0) {
        Write-Host "   ⚠️  Nessun modello trovato!" -ForegroundColor Yellow
        
        $response = Read-Host "`n   Vuoi scaricare '$Model'? (s/n)"
        if ($response -eq 's') {
            $installed = Install-Model -ModelName $Model
            if (-not $installed) {
                exit 1
            }
        } else {
            Write-Host "   💡 Scarica manualmente: ollama pull $Model`n" -ForegroundColor Yellow
        }
    } else {
        foreach ($m in $models) {
            Write-Host "   ✅ $m" -ForegroundColor Green
        }
    }
    
    # Test dipendenze Python
    $depsOk = Test-PythonDependencies
    
    if (-not $depsOk) {
        if ($InstallDeps) {
            $installed = Install-Dependencies
            if (-not $installed) {
                exit 1
            }
        } else {
            $response = Read-Host "`n   Vuoi installarle ora? (s/n)"
            if ($response -eq 's') {
                $installed = Install-Dependencies
                if (-not $installed) {
                    exit 1
                }
            } else {
                Write-Host "   💡 Installa manualmente: pip install -r requirements.txt`n" -ForegroundColor Yellow
                exit 1
            }
        }
    }
}

# Avvio applicazione
Write-Host "`n====================================================================" -ForegroundColor Cyan
Write-Host "✅ Sistema pronto!" -ForegroundColor Green
Write-Host "====================================================================" -ForegroundColor Cyan
Write-Host ""

Write-Host "🚀 Avvio Omni Eye AI...`n" -ForegroundColor Green

# Avvia l'applicazione
try {
    python start.py
} catch {
    Write-Host "`n❌ Errore avvio applicazione" -ForegroundColor Red
    Write-Host "   Dettagli: $_" -ForegroundColor Gray
}

Write-Host "`n👋 Arrivederci!`n" -ForegroundColor Cyan
