# Stable Deep Reinforcement Learning via Isotropic Gaussian Representations

This repository contains code to reproduce the experimental results from the paper:
Stable Deep Reinforcement Learning via Isotropic Gaussian Representations

---

## Installation

### Clone the repository
```bash
git clone https://github.com/asahebpa/IsoGaussian-DRL.git
cd IsoGaussian-DRL
```

### Create and activate the environment

```bash
conda create -n isogaussian-drl python=3.9 -y
conda activate isogaussian-drl
```

### Install dependencies

```bash
pip install -r requirements.txt
pip install -r requirements_atari.txt
```
---

## Codebase Origin and Attribution

The **main body of this codebase is cloned and adapted from**:

**Stable Deep Reinforcement Learning at Scale**
[https://github.com/roger-creus/stable-deep-rl-at-scale](https://github.com/roger-creus/stable-deep-rl-at-scale)

---

## Citation

If you use this repository in your work, please cite both the original codebase and this project.

### Original codebase and corresponding paper

```bibtex
@misc{creus2023stabledrl,
  title={Stable Deep Reinforcement Learning at Scale},
  author={Creus-Costa, Roger and others},
  year={2023},
  url={https://github.com/roger-creus/stable-deep-rl-at-scale}
}

@article{castanyer2025stable,
  title={Stable Gradients for Stable Learning at Scale in Deep Reinforcement Learning},
  author={Castanyer, Roger Creus and Obando-Ceron, Johan and Li, Lu and Bacon, Pierre-Luc and Berseth, Glen and Courville, Aaron and Castro, Pablo Samuel},
  journal={arXiv preprint arXiv:2506.15544},
  year={2025}
}
```

### This repository

```bibtex
@misc{sahebpasand2026isogaussian,
  title={Stable Deep Reinforcement Learning via Isotropic Gaussian Representations},
  author={Sahebpasand, Ali},
  year={2026},
  url={https://github.com/asahebpa/IsoGaussian-DRL}
}
```

