@echo off
echo ===================================
echo OT-CFM Style Transfer - Quick Test
echo ===================================
echo.

echo [1/3] Testing Model Architecture...
python test_model.py
if errorlevel 1 (
    echo.
    echo ERROR: Model test failed!
    pause
    exit /b 1
)

echo.
echo ===================================
echo Model test completed successfully!
echo ===================================
echo.
echo Next steps:
echo 1. Prepare your data (see USAGE_GUIDE.md)
echo 2. Run: python preprocess_latents.py
echo 3. Run: python train.py
echo 4. Run: python inference.py --checkpoint checkpoints/stage1_epochXX.pt --input test.jpg
echo.
pause
