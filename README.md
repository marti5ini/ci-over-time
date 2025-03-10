# A Practical Approach to Causal Inference over Time

[![Build Status](https://img.shields.io/badge/Build-Passing-brightgreen)](https://github.com/marti5ini/ci-over-time/actions) [![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)

This repository contains the implementation of the work [A Practical Approach to Causal Inference over Time](https://arxiv.org/abs/2410.10502), where we present a framework for estimating causal effects over time in dynamical systems. The proposed **causal VAR framework** allows us to perform causal inference over time from observational time series data.  
Our experiments on synthetic and real-world datasets show that the proposed framework achieves strong performance in terms of observational forecasting while enabling accurate estimation of the causal effect of interventions on dynamical systems. 

---

## Quick Start

Use the following Jupyter notebooks to reproduce the results presented in the paper:

- **[Experiments on Additive Interventions](https://github.com/marti5ini/ci-over-time/blob/main/experiments/interventional_forecasting_additive.py)**  
- **[Experiments on Forcing Interventions](https://github.com/marti5ini/ci-over-time/blob/main/experiments/interventional_forecasting_forcing.py)**  

These notebooks showcase how to evaluate the causal effects of interventions over time using the provided datasets and models.

---

## Setup

### Requirements

- Python >= 3.9
- Libraries listed in `requirements.txt`.

Additional libraries might be required for optional functionalities.

### Installation

Clone the repository and set up the environment:

```bash
git clone https://github.com/marti5ini/ci-over-time.git
cd ci-over-time
```
It is recommended to use a virtual environment:

```
python3 -m venv ./venv  # Optional but recommended
source ./venv/bin/activate
pip install -r requirements.txt
```

# Citation

If you use `our causal VAR framework` in your research, please cite our paper:

```
@inproceedings{
cinquini2025ciovertime,
title={A Practical Approach to Causal Inference over Time},
author={Cinquini, Martina, Beretta Isacco, Ruggieri, Salvatore and Valera, Isabel},
booktitle={The 39th Annual AAAI Conference on Artificial Intelligence},
year={2025},
url={https://openreview.net/forum?id=2EBCWWS0Me}
}

