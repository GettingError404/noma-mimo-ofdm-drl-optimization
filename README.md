# Deep Reinforcement Learning for NOMA-MIMO-OFDM Resource Allocation

This repository contains the implementation of a TD3-based deep reinforcement 
learning approach for joint power allocation and user pairing in NOMA-MIMO-OFDM systems.

## 🎯 Key Features

- Joint optimization of spectral efficiency (SE) and energy efficiency (EE)
- TD3 agent with continuous action space
- 3 baseline algorithms for comparison
- Realistic 3GPP UMi channel model
- Complete reproducible framework

## 📊 Main Results

- **SE Improvement**: 24.27% over best baseline
- **EE Improvement**: 24.27% over best baseline
- **Convergence**: Stable after ~500 episodes

## 🚀 Quick Start

### Installation
```bash
pip install -r requirements.txt
```

### Training
```bash
python experiments/train_drl.py
```

### Evaluation
```bash
python experiments/evaluate_baselines.py
```

## 📁 Repository Structure

- `configs/` - System parameters and configuration
- `src/environment/` - NOMA-MIMO-OFDM Gym environment
- `src/baselines/` - Baseline algorithms
- `src/drl/` - TD3 agent implementation
- `experiments/` - Training and evaluation scripts
- `results/` - Generated figures and tables

## 📖 Citation

If you use this code, please cite:
```bibtex
@article{yourname2025drl,
  title={Deep Reinforcement Learning for Joint Power Allocation and User Pairing in NOMA-MIMO-OFDM Systems},
  author={Bashid Tagala},
  journal={Under Review},
  year={2025}
}
```

## 📝 License

MIT License - see LICENSE file

## 🤝 Contact

basittagala831@gmail.com
(Initial commit: DRL + NOMA-MIMO-OFDM framework)
