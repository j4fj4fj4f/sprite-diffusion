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
- [ ] Set up Python environment
- [ ] Create project structure
- [ ] Add configuration system

### Phase 2 — Dataset
- [ ] Collect 64×64 sprite dataset
- [ ] Remove duplicates
- [ ] Verify image sizes
- [ ] Implement preprocessing
- [ ] Implement augmentation
- [ ] Build PyTorch Dataset

### Phase 3 — Diffusion Model
- [ ] Implement U-Net
- [ ] Implement forward diffusion
- [ ] Implement noise scheduler
- [ ] Implement reverse diffusion
- [ ] Implement sampling

### Phase 4 — Training
- [ ] Training loop
- [ ] TensorBoard logging
- [ ] Checkpoint saving
- [ ] EMA
- [ ] Validation sampling

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