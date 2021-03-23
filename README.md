# GLOW and WaveletFlow

This repo contains the GLOW and WaveletFlow Normalizing Flows.
Glow has been implemented but there is still development undergoing to port the original TensorFlow WaveletFlow code to PyTorch.

To run GLOW, run train_glow.py in the models respective directory. You can edit parameters in the general parameters.py file that share parameters between GLOW and WaveletFlow as well as the parameters.py file in the specific sub directory of the model. Inference.py is specifically for trained models of GLOW. Don't hesitate to contact me if anything is unclear since I did not intend to publicize this repo.

This repo is inspired from https://github.com/y0ast/Glow-PyTorch and contains some additional features such as masking strategies and plotting. This repo has been built with the intention to be expanded with other flows hence it might seem slightly over-engineered. The WaveletFlow model (https://arxiv.org/abs/2010.13821.pdf) does not work completely. Due to deadline issues I have had to halt this project.

Discord: Vzer#7201

Website: www.amaanv.com
