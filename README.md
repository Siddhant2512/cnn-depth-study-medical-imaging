# CNN Architectural Depth Study on PatchCamelyon

## Research Question
How does convolutional network depth affect classification performance, training stability, and computational efficiency on medical image classification tasks?

## Dataset
- **PatchCamelyon (PCam)**: 96×96 histopathology patches
- **Task**: Binary tumor classification
- **Size**: 262K train, 32K val, 32K test

## Architectures
- CNN_Depth4 (66K params)
- CNN_Depth8 (583K params)
- CNN_Depth16 (2.7M params)
- CNN_Depth24 (4.3M params)

## Results
Coming soon...

## Usage
```bash
# Setup
pip install -r requirements.txt

# Train models
python train.py --architecture all --seeds 42,43,44,45,46


