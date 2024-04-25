# SS-CMPE

Welcome to the README file of the official implementation **SS-CMPE** project! This repository contains the code implementation for the paper titled "Learning to Solve the Constrained Most Probable Explanation Task".

[paper](https://arxiv.org/abs/2201.08377](https://proceedings.mlr.press/v238/arya24b/arya24b.pdf)

## Setup and Installation 

To get started with the project, follow these steps:

1. Create a new conda environment and install the required dependencies by running the following command:

   ```sh
   conda create --name sscmpe --file requirements.txt --channel pytorch --channel nvidia --channel conda-forge
   ```

2. Activate the conda environment:

   ```sh
   conda activate sscmpe
   ```

## Experiments

This repository provides README files for specific sets of experiments in the following folders:

- **Tractable Probabilistic Circuits and High Tree-Width Markov Networks (ssl_adv)**:
  This folder contains the README file for experiments related to Tractable Probabilistic Circuits and High Tree-Width Markov Networks. Please navigate to the `ssl_adv` folder to access the specific README instructions.

- **Adversarial Example Generation (ssl_pgm)**:
  This folder contains the README file for experiments related to Adversarial Example Generation. Please navigate to the `ssl_pgm` folder to access the specific README instructions.

Please refer to the respective README files in the mentioned folders for detailed instructions on running the experiments and utilizing the provided code.

## Citation

If this work is helpful in your research, please consider starring :star: us and citing:  

```bibtex
@inproceedings{arya_2024_solveconstraineda,
  title = {Learning to {{Solve}} the {{Constrained Most Probable Explanation Task}} in {{Probabilistic Graphical Models}}},
  booktitle = {Proceedings of {{The}} 27th {{International Conference}} on {{Artificial Intelligence}} and {{Statistics}}},
  author = {Arya, Shivvrat and Rahman, Tahrima and Gogate, Vibhav},
  year = {2024},
  month = apr,
  pages = {2791--2799},
  publisher = {PMLR},
  issn = {2640-3498},
  urldate = {2024-04-21},
}

```

