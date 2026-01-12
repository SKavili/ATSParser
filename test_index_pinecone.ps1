# Simple script to test the index-pinecone API

Write-Host "Testing /api/v1/index-pinecone API..." -ForegroundColor Green
Write-Host ""

# Test with 1 resume
$result = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/index-pinecone?limit=1" -Method POST

Write-Host "Result:" -ForegroundColor Yellow
$result | Format-List

Write-Host ""
Write-Host "Done!" -ForegroundColor Green

