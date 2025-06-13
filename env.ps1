# Get the directory where the script is located
$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition

# Define relative paths based on script location
$env:DATASET_PATH = Join-Path $ScriptDir "datasets\"
$env:STORE_LOCATION = Join-Path $ScriptDir "results\"

# Print the environment variable values
Write-Host "Environment variables set:"
Write-Host "DATASET_PATH=$env:DATASET_PATH"
Write-Host "STORE_LOCATION=$env:STORE_LOCATION"
