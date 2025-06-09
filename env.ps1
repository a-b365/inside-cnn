# Set environment variables for the current session
$env:ORIGINAL_DATA_PATH = "D:\inside-cnn\datasets\3702_left_knee.nii.gz"
$env:MASK_DATA_PATH = "D:\inside-cnn\datasets\3702_left_knee_mask_final.nii.gz"
$env:STORE_LOCATION = "D:\inside-cnn\results\"

Write-Host "Environment variables set:"
Write-Host "ORIGINAL_DATA_PATH=$env:ORIGINAL_DATA_PATH"
Write-Host "MASK_DATA_PATH=$env:MASK_DATA_PATH"
Write-Host "STORE_LOCATION=$env:STORE_LOCATION"