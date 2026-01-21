#!/bin/bash

echo "========================================"
echo "OT-CFM Style Transfer - Quick Start"
echo "========================================"
echo ""

# 颜色定义
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 检查 Python
if ! command -v python &> /dev/null; then
    echo -e "${RED}Error: Python not found!${NC}"
    exit 1
fi

echo -e "${GREEN}[1/4] Checking dependencies...${NC}"
python -c "import torch; print(f'PyTorch: {torch.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: PyTorch not installed. Installing...${NC}"
    pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
fi

python -c "import diffusers; print(f'Diffusers: {diffusers.__version__}')" 2>/dev/null
if [ $? -ne 0 ]; then
    echo -e "${YELLOW}Warning: Diffusers not installed. Installing...${NC}"
    pip install diffusers transformers accelerate
fi

echo ""
echo -e "${GREEN}[2/4] Testing model architecture...${NC}"
python test_model.py
if [ $? -ne 0 ]; then
    echo -e "${RED}Error: Model test failed!${NC}"
    exit 1
fi

echo ""
echo -e "${GREEN}[3/4] Checking data...${NC}"
if [ ! -d "$(python -c "import json; print(json.load(open('config.json'))['data']['data_root'])")" ]; then
    echo -e "${YELLOW}Warning: Processed data not found.${NC}"
    echo -e "${YELLOW}Please run: python preprocess_latents.py${NC}"
else
    echo -e "${GREEN}✓ Data directory exists${NC}"
fi

echo ""
echo -e "${GREEN}[4/4] Setup complete!${NC}"
echo ""
echo "========================================"
echo "Next steps:"
echo "========================================"
echo "1. Prepare your data:"
echo "   - Organize images in raw_data_root/style_0, style_1, ..."
echo "   - Run: python preprocess_latents.py"
echo ""
echo "2. Train the model:"
echo "   - Run: python train.py"
echo ""
echo "3. Run inference:"
echo "   - Run: python inference.py --checkpoint checkpoints/stage1_epoch200.pt --input test.jpg --source_style 0 --target_style 1 --output result.png"
echo ""
echo "For detailed instructions, see USAGE_GUIDE.md"
echo "========================================"
