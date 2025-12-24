# PowerShell script to start exo in Docker on Windows

Write-Host "Building exo Docker image..." -ForegroundColor Green
docker-compose build

if ($LASTEXITCODE -eq 0) {
    Write-Host "Starting exo container..." -ForegroundColor Green
    docker-compose up -d
    
    Write-Host ""
    Write-Host "exo is starting up..." -ForegroundColor Yellow
    Write-Host "Dashboard will be available at: http://localhost:52415" -ForegroundColor Cyan
    Write-Host ""
    Write-Host "To view logs, run: docker-compose logs -f" -ForegroundColor Gray
    Write-Host "To stop, run: docker-compose down" -ForegroundColor Gray
} else {
    Write-Host "Build failed!" -ForegroundColor Red
    exit 1
}

