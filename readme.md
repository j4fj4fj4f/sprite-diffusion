# Pixel Diffusion

A diffusion model for generating 64×64 pixel art character sprites from scratch in PyTorch.

## Goal

The long-term goal of this project is to build a controllable diffusion model capable of generating pixel art characters from user-defined attributes, such as:

- Character class (Mage, Knight, Archer, ...)
- Hair color
- Eye color
- Clothing
- Weapons
- Accessories

The project starts with unconditional sprite generation before gradually introducing conditioning and text guidance.

---

## Roadmap

### Phase 1 — Project Setup
- [x] Create repository
- [x] Set up Python environment
- [x] Create project structure
- [x] Add configuration system

### Phase 2 — Dataset
- [x] Collect 64×64 sprite dataset
- [x] Remove duplicates
- [x] Verify image sizes
- [ ] Implement preprocessing
- [ ] Implement augmentation
- [x] Build PyTorch Dataset

### Phase 3 — Diffusion Model
- [x] Implement U-Net
- [x] Implement forward diffusion
- [x] Implement noise scheduler
- [x] Implement reverse diffusion
- [x] Implement sampling

### Phase 4 — Training
- [x] Training loop
- [ ] TensorBoard logging
- [x] Checkpoint saving
- [x] EMA
- [x] Validation sampling

### Phase 5 — Conditional Generation
- [ ] Class conditioning
- [ ] Attribute conditioning
- [ ] Conditional sampling

### Phase 6 — Text Conditioning
- [ ] Prompt encoder
- [ ] Text-conditioned diffusion

---

## Tech Stack

- Python 3.11
- PyTorch
- TorchVision
- Pillow
- NumPy
- TensorBoard

---

## Current Status

🚧 Project initialization