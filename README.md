# Thought Anchors: Which LLM Reasoning Steps Matter?

This repository implements the experiments outlined in the paper ["Thought Anchors: Which LLM Reasoning Steps Matter?"](https://arxiv.org/pdf/2506.19143v1) by Bogdan et al.

## Overview

Thought Anchors analyzes reasoning large language models at the sentence level to identify critical reasoning steps that guide the rest of the reasoning trace. The paper presents three complementary attribution methods:

1. **Black-box Resampling**: Measures counterfactual importance by comparing final answers across rollouts conditioned on different sentences
2. **Receiver Heads Analysis**: Identifies "broadcasting" sentences that receive disproportionate attention via "receiver" attention heads
3. **Attention Suppression**: Measures causal dependencies between sentence pairs by suppressing attention and measuring effects

## Methods

### 1. Black-box Resampling Method
- Resamples reasoning traces from the start of each sentence
- Quantifies impact of each sentence on final answer likelihood
- Distinguishes planning sentences from computation sentences
- Measures counterfactual importance through 100 rollouts per sentence

### 2. Receiver Heads Analysis
- Identifies "receiver" attention heads that narrow attention toward specific past sentences
- Measures "broadcasting" sentences that receive disproportionate attention
- Compares attention patterns between reasoning and base models
- Provides mechanistic measure of sentence importance

### 3. Attention Suppression Method
- Masks attention to specific sentences from subsequent tokens
- Measures effect on subsequent token logits (KL divergence)
- Maps direct causal effects between sentence pairs
- Creates directed acyclic graphs of reasoning dependencies

## Sentence Taxonomy

The implementation uses an 8-category taxonomy for reasoning functions:

1. **Problem Setup**: Parsing or rephrasing the problem
2. **Plan Generation**: Stating or deciding on a plan of action, meta-reasoning
3. **Fact Retrieval**: Recalling facts, formulas, problem details without computation
4. **Active Computation**: Algebra, calculations, or other manipulations toward the answer
5. **Uncertainty Management**: Expressing confusion, re-evaluating, including backtracking
6. **Result Consolidation**: Aggregating intermediate results, summarizing, or preparing
7. **Self Checking**: Verifying previous steps, checking calculations, and re-confirmations
8. **Final Answer Emission**: Explicitly stating the final answer

## Setup

### Prerequisites
- Python 3.8+
- PyTorch
- Transformers library
- NumPy
- Pandas
- Matplotlib/Plotly for visualization

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd thought_anchors

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
cp .env.example .env
# Edit .env with your model paths and API keys
```

### Configuration

1. **Model Setup**: Configure your reasoning model in `config/models.py`
2. **Dataset**: Place your reasoning traces dataset in `data/`
3. **Environment**: Set up API keys and model paths in `.env`

## Usage

### Running Experiments

```bash
# Black-box resampling experiments
python experiments/blackbox_resampling.py

# Receiver heads analysis
python experiments/receiver_heads.py

# Attention suppression experiments
python experiments/attention_suppression.py

# Full pipeline
python run_experiments.py
```

### Visualization

```bash
# Generate visualizations
python visualization/generate_plots.py

# Interactive dashboard (if implemented)
python dashboard/app.py
```

## Project Structure

```
thought_anchors/
├── README.md
├── requirements.txt
├── setup.py
├── .env.example
├── config/
│   ├── __init__.py
│   ├── models.py
│   └── experiments.py
├── data/
│   ├── raw/
│   ├── processed/
│   └── results/
├── src/
│   ├── __init__.py
│   ├── sentence_analysis/
│   │   ├── __init__.py
│   │   ├── taxonomy.py
│   │   └── labeling.py
│   ├── blackbox/
│   │   ├── __init__.py
│   │   ├── resampling.py
│   │   └── importance.py
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── receiver_heads.py
│   │   └── suppression.py
│   └── utils/
│       ├── __init__.py
│       ├── model_utils.py
│       └── data_utils.py
├── experiments/
│   ├── __init__.py
│   ├── blackbox_resampling.py
│   ├── receiver_heads.py
│   └── attention_suppression.py
├── visualization/
│   ├── __init__.py
│   ├── plots.py
│   └── dashboard.py
├── tests/
│   ├── __init__.py
│   ├── test_sentence_analysis.py
│   ├── test_blackbox.py
│   └── test_attention.py
└── notebooks/
    ├── exploration.ipynb
    └── results_analysis.ipynb
```

## Key Features

- **Modular Design**: Each method is implemented as a separate module
- **Configurable**: Easy to modify experiments and parameters
- **Reproducible**: All experiments are deterministic and documented
- **Extensible**: Easy to add new attribution methods
- **Visualization**: Built-in plotting and dashboard capabilities

## Citation

If you use this code in your research, please cite the original paper:

```bibtex
@article{bogdan2024thought,
  title={Thought Anchors: Which LLM Reasoning Steps Matter?},
  author={Bogdan, Paul C. and Macar, Uzay and Nanda, Neel and Conmy, Arthur},
  journal={arXiv preprint arXiv:2506.19143},
  year={2024}
}
```

**Note**: This repository is an implementation of the methods described in the above paper. When using this code, please cite the original paper authors, not the repository maintainers.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- Original paper authors for the research
- The open-source community for the tools and libraries used
- Contributors to this implementation

## Contact

For questions or issues, please open an issue on GitHub or contact the maintainers. 