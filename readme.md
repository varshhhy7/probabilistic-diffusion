# Probabilistic Diffusion

A from-scratch implementation of **Denoising Diffusion Probabilistic Models (DDPMs)** with a focus on probabilistic clarity, mathematical rigor, and interpretability.

## 🎯 Project Overview

This repository contains a single, well-documented Jupyter notebook that walks through implementing a DDPM from first principles. Rather than obscuring the mathematics behind production-grade libraries, this project emphasizes:

- **Clear mathematical derivations** of the forward and reverse diffusion processes
- **Explicit probabilistic modeling** with step-by-step explanations
- **Clean, interpretable PyTorch code** separated from mathematical exposition
- **Extensibility** as a foundation for experimenting with variants and improvements

Perfect for researchers, students, and practitioners seeking to understand or implement diffusion models.

## 📋 Table of Contents

- [Features](#-features)
- [Project Structure](#-project-structure)
- [Getting Started](#-getting-started)
- [Implementation Details](#-implementation-details)
- [Results & Experiments](#-results--experiments)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [Resources](#-resources)

## ✨ Features

### Core Implementation

- ✅ **Single Jupyter Notebook** - All code, explanations, and experiments in one readable document
- ✅ **From-Scratch DDPM** - No hidden abstractions; every component is explicitly implemented
- ✅ **Mathematical Clarity** - Full derivations of:
  - Forward diffusion schedule (β_t, α_t, ᾱ_t)
  - Closed-form q(x_t | x_0) and its role in training
  - Reverse denoising process parameterization p_θ(x_{t-1} | x_t)
  - Training objective and loss functions

### Code Organization

- ✅ **Modular Sections**:
  - Setup and utility functions
  - Forward diffusion process
  - Neural network architecture (ε_θ)
  - Training loop with loss computation
  - Sampling and generation pipeline
  - Visualization utilities

- ✅ **Clear Separation** between:
  - Mathematical exposition (equations, intuitions, proofs)
  - Implementation blocks (PyTorch code, configurations)
  - Experimental cells (training, inference, analysis)

### Extensibility

- ✅ **Multiple Noise Schedules** - Linear, cosine, and custom schedules
- ✅ **Alternative Parameterizations** - Noise prediction (ε), x_0 prediction, velocity prediction (v)
- ✅ **Flexible Architectures** - Easy swapping of network designs (simple MLP, UNet, attention-based)
- ✅ **Customizable Hyperparameters** - Dataset size, diffusion steps, learning rates, batch sizes, etc.

## 📁 Project Structure

```
probabilistic-diffusion/
├── Diffusion_probabilistic_model.ipynb    # Main notebook (all code & explanations)
├── 2006.11239v2 (1).pdf                   # Original DDPM paper
├── Diffusion_Models_Unlocked.pdf          # Reference material
└── readme.md                               # This file
```

**Note:** All code, experiments, derivations, and results live in `Diffusion_probabilistic_model.ipynb`. No external modules or scripts are required.

## 🚀 Getting Started

### Prerequisites

You can run this notebook locally or in a cloud environment (Google Colab recommended for GPU access).

**Recommended Software:**
- Python 3.9+
- Jupyter Notebook or JupyterLab
- PyTorch (with GPU support for faster training)
- NumPy, Matplotlib, tqdm

**Install Dependencies:**

```bash
# Core PyTorch installation (CPU)
pip install torch torchvision torchaudio

# GPU-enabled installation (CUDA 11.8)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Additional packages
pip install numpy matplotlib tqdm jupyter
```

**For Google Colab:**
Most dependencies are preinstalled. In the notebook, uncomment and run the setup cell if needed.

### Running the Notebook

#### Option 1: Local Execution

```bash
# Clone the repository
git clone https://github.com/varshhhy7/probabilistic-diffusion.git
cd probabilistic-diffusion

# Launch Jupyter
jupyter notebook

# Open Diffusion_probabilistic_model.ipynb in your browser
```

#### Option 2: Google Colab (Recommended)

1. Open [Google Colab](https://colab.research.google.com/)
2. Go to **File → Open notebook**
3. Paste the GitHub URL: `https://github.com/varshhhy7/probabilistic-diffusion`
4. Select and open `Diffusion_probabilistic_model.ipynb`
5. Run cells sequentially from top to bottom

**Note:** Colab provides free GPU (T4), making training significantly faster.

### Workflow

The notebook follows this workflow:

1. **Load/Generate Data** - Create or load a dataset (MNIST, toy images, etc.)
2. **Define Diffusion Schedule** - Set β_t values and compute α_t, ᾱ_t
3. **Build Network** - Define ε_θ(x_t, t) to predict noise
4. **Train Model** - Train using the diffusion objective (MSE loss)
5. **Generate Samples** - Run reverse process from pure Gaussian noise
6. **Visualize Results** - Display intermediate and final samples

You can modify hyperparameters directly in the notebook:
- Number of diffusion steps (T)
- Noise schedule type (linear, cosine, etc.)
- Learning rate and training epochs
- Batch size and dataset
- Network architecture (hidden layers, attention, etc.)

## 🔍 Implementation Details

### Forward Diffusion Process

The forward process gradually adds noise to data over T steps:

```
q(x_t | x_{t-1}) = N(x_t; √(1-β_t) x_{t-1}, β_t I)
```

Using the reparameterization trick, we derive the closed-form:

```
q(x_t | x_0) = N(x_t; √(ᾱ_t) x_0, (1-ᾱ_t) I)
```

This allows efficient one-shot sampling from any timestep during training.

### Model Architecture

The core component is a neural network ε_θ(x_t, t) that predicts the noise added at timestep t:

- **Input:** Noisy sample x_t (image) + timestep t (scalar)
- **Output:** Predicted noise ε_pred
- **Parameterization:** Simple MLP (easily replaceable with UNet, attention, etc.)
- **Timestep Encoding:** Sinusoidal position embeddings or learned embeddings

### Training Objective

The model is trained using the variational lower bound, which simplifies to:

```
L = E_{x_0, ε, t} [ ||ε - ε_θ(x_t, t)||^2 ]
```

Where:
- x_0 ~ data distribution
- ε ~ N(0, I) (random noise)
- t ~ Uniform(1, T) (random timestep)

### Reverse Process (Sampling)

Once trained, we generate new samples by iteratively denoising:

```
x_{t-1} ~ p_θ(x_{t-1} | x_t) = N(x_{t-1}; μ_θ(x_t, t), Σ_θ(x_t, t))
```

Starting from x_T ~ N(0, I), we run the process backwards to obtain x_0.

### Key Equations

| Component | Equation | Purpose |
|-----------|----------|----------|
| Schedule | β_t ∈ (0, 1) | Controls noise level at each step |
| Compound | α_t = 1 - β_t | Retention factor |
| Cumulative | ᾱ_t = ∏α_i | Jump-to-t efficiency |
| Forward | q(x_t\|x_0) = N(√ᾱ_t x_0, (1-ᾱ_t)I) | Closed-form forward |
| Loss | \|\|ε - ε_θ(x_t, t)\|\|^2 | Training objective |

## 📊 Results & Experiments

The notebook includes experiments on:

- **MNIST**: Simple digit generation as a sanity check
- **Custom Toy Data**: Synthetic 2D distributions to visualize diffusion
- **CelebA / CIFAR-10** (optional): Larger-scale image generation

**Expected Outcomes:**
- Model learns to reverse the forward diffusion process
- Generated samples become increasingly coherent as noise is removed
- Loss converges smoothly over training epochs

## 🗺️ Roadmap

Planned enhancements and extensions:

### Near-term
- [ ] Support multiple noise schedules (linear, cosine, polynomial)
- [ ] Add UNet architecture as alternative to MLP
- [ ] Implement attention mechanisms for better image quality
- [ ] Configuration cells for easy hyperparameter tuning
- [ ] Model checkpointing and loading
- [ ] Visualization of diffusion process (forward and reverse)

### Medium-term
- [ ] Conditional diffusion (class-conditional generation)
- [ ] Alternative parameterizations (x_0 prediction, velocity)
- [ ] Latent diffusion (operate in embedding space)
- [ ] Classifier-free guidance
- [ ] Unit tests and validation checks

### Long-term
- [ ] Text-to-image diffusion (CLIP guidance)
- [ ] Super-resolution and inpainting variants
- [ ] Diffusion models for other modalities (audio, video)
- [ ] Quantization and efficiency improvements
- [ ] Multi-GPU training support

## 📚 How to Use This Repository

### For Learning

**Goal:** Understand DDPMs deeply

1. Read the mathematical exposition cells first
2. Study the equations and intuitions
3. Trace through the code cells side-by-side
4. Experiment with hyperparameters and visualize the effects
5. Read the research papers for deeper context

### For Research/Development

**Goal:** Build upon this implementation

1. Use as a **reference implementation** in your own projects
2. Port individual components to your codebase
3. Experiment with:
   - Different noise schedules
   - Alternative architectures (UNet, Transformers)
   - Different datasets and domains
   - Novel parameterizations and loss weightings

### For Production

**Goal:** Deploy generative models

1. Extract trained model weights
2. Optimize architecture for inference speed
3. Use model quantization and distillation
4. Integrate into your application
5. Monitor generation quality and sample diversity

## 🤝 Contributing

Contributions are highly welcome! Areas of interest:

- **Documentation**: Improve mathematical explanations, add diagrams
- **Visualizations**: Create plots that clarify the diffusion process
- **Code Quality**: Better abstractions, cleaner implementations
- **Extensions**: New architectures, datasets, loss functions
- **Optimization**: Faster training, reduced memory usage
- **Bug Fixes**: Report and fix issues

### Contributing Guidelines

1. **Fork** the repository
2. **Create a feature branch**: `git checkout -b feature/your-feature`
3. **Make changes** and ensure the notebook runs cleanly top-to-bottom
4. **Test thoroughly** - run cells multiple times, verify outputs
5. **Commit with clear messages**: `git commit -m "Add feature: description"`
6. **Push and create a Pull Request** with:
   - Clear description of changes
   - Motivation and benefits
   - Example outputs or visualizations
   - Any new hyperparameters or dependencies

## 📖 Resources

### Papers

- **[Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239)** - Ho et al., 2020 (original DDPM)
- **[Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233)** - Dhariwal & Nichol, 2021
- **[Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598)** - Ho & Salimans, 2022
- **[High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752)** - Rombach et al., 2022

### Textbooks & Tutorials

- [Hugging Face Diffusion Course](https://huggingface.co/course/chapter1/1) - Interactive diffusion learning
- [Lil'Log on Diffusion Models](https://lilianweng.github.io/posts/2021-07-11-diffusion-models/) - Comprehensive overview
- [OpenAI Diffusion Models Intro](https://openai.com/blog/diffusion-models/) - Intuitive explanations

### Reference Implementations

- [CompVis/latent-diffusion](https://github.com/CompVis/latent-diffusion) - Stable Diffusion
- [openai/guided-diffusion](https://github.com/openai/guided-diffusion) - OpenAI's guided diffusion
- [lucidrains/denoising-diffusion-pytorch](https://github.com/lucidrains/denoising-diffusion-pytorch) - Minimalist PyTorch DDPM

## 📝 License

This project is open source and available under the MIT License.

## 💬 Questions & Discussions

Have questions or ideas? Feel free to:

- **Open an Issue** on GitHub for bugs or feature requests
- **Start a Discussion** for questions about the implementation
- **Create a Pull Request** with suggestions or improvements

---

**Last Updated:** January 2026

If you found this helpful, consider leaving a ⭐ on GitHub!
