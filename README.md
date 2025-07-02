# Wave-GAN: Deep Learning for Nonlinear Wave-Structure Interaction Prediction

This repository contains the TensorFlow implementation of the Wave-GAN framework introduced in the paper:

**Wave-GAN: A deep learning approach for the prediction of nonlinear wave-structure interactions**  
Blanca Peña et al., *Coastal Engineering*, 2021  
[https://doi.org/10.1016/j.cpc.2021.107440](https://www.sciencedirect.com/science/article/abs/pii/S0378383921000624)

---

## Overview

Wave-GAN leverages Generative Adversarial Networks (GANs), specifically a Pix2Pix-inspired conditional GAN architecture, to predict complex nonlinear interactions between ocean waves and offshore structures. This approach provides a computationally efficient alternative to traditional physics-based simulations, enabling fast and accurate predictions for engineering applications.

This code implements the core image-to-image translation pipeline underlying Wave-GAN:

- Modified U-Net generator with skip connections  
- PatchGAN discriminator for local adversarial feedback  
- Combined L1 and adversarial losses for robust training  
- Data preprocessing with augmentation, normalization, and batching  
- Training loop with checkpointing and TensorBoard support  

---

## Dataset Format

Input data should be arranged as paired images concatenated side-by-side (input | target), organized under the following folders:
