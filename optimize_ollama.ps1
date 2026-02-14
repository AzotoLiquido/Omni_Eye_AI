# ─────────────────────────────────────────────────────────────
# Omni Eye AI — Ottimizzazione Ollama per RTX 2080 (8GB VRAM)
# ─────────────────────────────────────────────────────────────
#
# Questo script imposta le variabili d'ambiente di Ollama
# per prestazioni ottimali sulla GPU NVIDIA.
#
# Uso:    .\optimize_ollama.ps1
# Nota:   Richiede riavvio di Ollama dopo l'esecuzione
# ─────────────────────────────────────────────────────────────

Write-Host ""
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host "  Omni Eye AI — Ottimizzazione Ollama" -ForegroundColor Cyan
Write-Host "  GPU: RTX 2080 (8GB VRAM, Compute 7.5)" -ForegroundColor Cyan
Write-Host "======================================================" -ForegroundColor Cyan
Write-Host ""

# ── 1. Flash Attention ──────────────────────────────────────
# Accelera il calcolo dell'attenzione del 30-50%
# Richiede compute capability >= 7.0 (RTX 2080 = 7.5 ✓)
[System.Environment]::SetEnvironmentVariable('OLLAMA_FLASH_ATTENTION', '1', 'User')
$env:OLLAMA_FLASH_ATTENTION = '1'
Write-Host "[OK] OLLAMA_FLASH_ATTENTION = 1  (30-50% piu' veloce)" -ForegroundColor Green

# ── 2. KV Cache quantizzata ────────────────────────────────
# Riduce l'uso di VRAM per il contesto (~50% risparmio KV cache)
# Permette contesti più grandi senza saturare la VRAM
[System.Environment]::SetEnvironmentVariable('OLLAMA_KV_CACHE_TYPE', 'q8_0', 'User')
$env:OLLAMA_KV_CACHE_TYPE = 'q8_0'
Write-Host "[OK] OLLAMA_KV_CACHE_TYPE = q8_0  (50% meno VRAM per contesto)" -ForegroundColor Green

# ── 3. Max modelli caricati ────────────────────────────────
# Con 8GB VRAM, conviene tenere 1 solo modello alla volta
# Evita swap GPU e rallentamenti
[System.Environment]::SetEnvironmentVariable('OLLAMA_MAX_LOADED_MODELS', '1', 'User')
$env:OLLAMA_MAX_LOADED_MODELS = '1'
Write-Host "[OK] OLLAMA_MAX_LOADED_MODELS = 1  (evita swap GPU)" -ForegroundColor Green

# ── 4. Parallelismo ───────────────────────────────────────
# 1 richiesta alla volta per massimizzare velocità singola inferenza
[System.Environment]::SetEnvironmentVariable('OLLAMA_NUM_PARALLEL', '1', 'User')
$env:OLLAMA_NUM_PARALLEL = '1'
Write-Host "[OK] OLLAMA_NUM_PARALLEL = 1  (max velocita' singola)" -ForegroundColor Green

Write-Host ""
Write-Host "======================================================" -ForegroundColor Yellow
Write-Host "  Variabili impostate permanentemente (utente)." -ForegroundColor Yellow
Write-Host "  Riavvia Ollama per applicare le modifiche." -ForegroundColor Yellow
Write-Host "======================================================" -ForegroundColor Yellow
Write-Host ""

# ── Riavvio opzionale ──────────────────────────────────────
$restart = Read-Host "Vuoi riavviare Ollama ora? (s/N)"
if ($restart -eq 's' -or $restart -eq 'S') {
    Write-Host ""
    Write-Host "Arresto Ollama..." -ForegroundColor Cyan
    Get-Process -Name "ollama*" -ErrorAction SilentlyContinue | Stop-Process -Force
    Start-Sleep -Seconds 3

    Write-Host "Avvio Ollama con le nuove impostazioni..." -ForegroundColor Cyan

    # Cerca l'eseguibile Ollama
    $ollamaPath = Get-Command ollama -ErrorAction SilentlyContinue | Select-Object -ExpandProperty Source
    if ($ollamaPath) {
        $ollamaDir = Split-Path $ollamaPath -Parent
        $ollamaApp = Join-Path $ollamaDir "ollama app.exe"
        if (Test-Path $ollamaApp) {
            Start-Process $ollamaApp -WindowStyle Hidden
        } else {
            Start-Process "ollama" -ArgumentList "serve" -WindowStyle Hidden
        }
    } else {
        Write-Host "[!] Ollama non trovato nel PATH. Avvialo manualmente." -ForegroundColor Yellow
    }

    Start-Sleep -Seconds 4

    # Verifica
    try {
        $null = Invoke-WebRequest -Uri "http://localhost:11434/api/tags" -UseBasicParsing -TimeoutSec 5
        Write-Host "[OK] Ollama riavviato con successo!" -ForegroundColor Green
    }
    catch {
        Write-Host "[!] Ollama potrebbe impiegare qualche secondo ad avviarsi." -ForegroundColor Yellow
        Write-Host "    Verifica con: ollama list" -ForegroundColor Yellow
    }
}

Write-Host ""
Write-Host "Fatto! Puoi verificare le ottimizzazioni con:" -ForegroundColor Cyan
Write-Host "  python train.py info" -ForegroundColor White
Write-Host "  python train.py benchmark" -ForegroundColor White
Write-Host ""
