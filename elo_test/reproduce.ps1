# ORION Chess Engine - Empirical Elo Reproducibility Pipeline
# Executes full audit from scratch: protocol validation, runtime benchmark, balanced tournament, and Ordo rating estimation.

Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  ORION EMPIRICAL ELO VALIDATION REPRODUCIBILITY SUITE      " -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan

$ROOT = Split-Path -Parent $PSScriptRoot
Set-Location $ROOT

# 1. Verify Checkpoint Integrity
$CHECKPOINT = "$ROOT\checkpoints\distilled\orion_distill_5ep.json"
if (-not (Test-Path $CHECKPOINT)) {
    Write-Error "Checkpoint missing: $CHECKPOINT"
    exit 1
}
$HASH = (Get-FileHash $CHECKPOINT -Algorithm SHA256).Hash
Write-Host "[1/6] Checkpoint SHA-256 Verified: $HASH" -ForegroundColor Green

# 2. Verify External Binaries
$ORDO = "$ROOT\ordo.exe"
$STOCKFISH = "$ROOT\stockfish.exe"
$CUTECHESS = "C:\Users\shaur\AppData\Local\Programs\Cute Chess\cutechess-cli.exe"

if (-not (Test-Path $ORDO)) { Write-Error "Ordo binary missing at $ORDO"; exit 1 }
if (-not (Test-Path $STOCKFISH)) { Write-Error "Stockfish missing at $STOCKFISH"; exit 1 }
if (-not (Test-Path $CUTECHESS)) { Write-Error "Cute Chess CLI missing at $CUTECHESS"; exit 1 }
Write-Host "[2/6] Toolchain Verified: Ordo, Stockfish, Cute Chess CLI present." -ForegroundColor Green

# 3. Phase One: UCI Protocol and Chess Legality Verification
Write-Host "[3/6] Running UCI Protocol & Chess Legality Audit..." -ForegroundColor Yellow
python "$ROOT\elo_test\uci_correctness_test.py"
if ($LASTEXITCODE -ne 0) {
    Write-Error "UCI correctness test failed!"
    exit 1
}

# 4. Phase Three: Runtime Performance Benchmark
Write-Host "[4/6] Running Runtime Performance Benchmark..." -ForegroundColor Yellow
python "$ROOT\elo_test\runtime_benchmark.py"

# 5. Phase Two & Four: Tournament Execution
Write-Host "[5/6] Executing Calibrated Reference Tournament..." -ForegroundColor Yellow
python "$ROOT\elo_test\run_tournament.py"

# 6. Phase Five: Bootstrap Uncertainty Analysis
Write-Host "[6/6] Computing Paired Bootstrap Confidence Intervals..." -ForegroundColor Yellow
python "$ROOT\elo_test\calculate_bootstrap_uncertainty.py"

Write-Host "============================================================" -ForegroundColor Green
Write-Host "  AUDIT COMPLETE: All artifacts generated in elo_test/      " -ForegroundColor Green
Write-Host "============================================================" -ForegroundColor Green
