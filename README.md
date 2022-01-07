# ParticleGridMapper

[![Stable](https://img.shields.io/badge/docs-stable-blue.svg)](https://huchiayu.github.io/ParticleGridMapper.jl/stable)
[![Dev](https://img.shields.io/badge/docs-dev-blue.svg)](https://huchiayu.github.io/ParticleGridMapper.jl/dev)
[![Build Status](https://github.com/huchiayu/ParticleGridMapper.jl/workflows/CI/badge.svg)](https://github.com/huchiayu/ParticleGridMapper.jl/actions)
[![Coverage](https://codecov.io/gh/huchiayu/ParticleGridMapper.jl/branch/master/graph/badge.svg)](https://codecov.io/gh/huchiayu/ParticleGridMapper.jl)


```ParticleGridMapper.jl``` interpolates particle data onto an adaptive mesh where each cell contains no more than one particle.

![vis_amr](https://user-images.githubusercontent.com/23061774/137218103-79a368f5-1de1-42a0-836a-3530e2a03ffa.png)


# Examples

 - [examples/example_cloud.jl](https://github.com/huchiayu/ParticleGridMapper.jl/blob/master/examples/example_cloud.jl) demonstrates the usage of five different interpolation schemes: (1) Cartesian mesh with nearest gid point (NGP); (2) Cartesian mesh with smoothed-particle hydrodynamics (SPH); (3) NGP on an adaptive mesh; (4) SPH on an adaptive mesh; (5) Meshless finite mass (MFM) on an adaptive mesh.

## 2:1 balance
![2to1balance_medium](https://user-images.githubusercontent.com/23061774/137220920-c9c07570-d658-4fb8-b34c-2c305196c67b.gif)


# Author
Chia-Yu Hu @ Max Planck Institute for Extraterrestrial Physics 
(cyhu.astro@gmail.com)
