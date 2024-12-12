# README: Dynamic Movie Recommendations

## Overview
Dynamic Movie Recommendations explores online recommendation algorithms such as A/B Testing, Epsilon-Greedy, Thompson Sampling, Softmax, and UCB. The project compares their performance in recommending movies to users based on historical data. Its main contributions include:

- Simulating online recommendation environments.
- Implementing and comparing multiple bandit algorithms.
- Providing insights into optimal recommendation strategies.

## Environment Setup

### Programming Language
- Python 3.8+

### Dependencies
Install dependencies using the `requirements.txt` file:
```bash
pip install -r requirements.txt
```
#### requirements.txt
```
numpy==1.21.2
pandas==1.3.3
matplotlib==3.4.3
seaborn==0.11.2
tqdm==4.62.3
```

## Data

### Source
This project uses the MovieLens 100k dataset. Ensure the `data/raw/` directory contains:
- `u.item` (movie metadata)
- `u.data` (user ratings)

### Preprocessing
Run the `0_data_exploration.ipynb` notebook to:
1. Process raw data into usable formats.
2. Focus on the 10 movies with the most ratings.
3. Transform ratings into binary rewards (`like` or `dislike`).
4. Save processed data to `data/top-n-movies_user-ratings.csv`.

## Running the Code

### Simulations
1. Open and run the `1_bandit_simulator.ipynb` notebook to simulate:
   - A/B Testing
   - Epsilon-Greedy (ε=0.05, 0.10)
   - Thompson Sampling
   - Softmax (τ=0.1)
   - UCB (c=1.0)

   Output files are saved to the `output/` directory.

2. Open and run `2_result_analysis.ipynb` to analyze results.
3. Open and run `3_metrics.ipynb` to visualize metrics such as cumulative rewards and regret.

### Sample Run
To run the code, simply open the relevant `.ipynb` file in Jupyter Notebook or JupyterLab and click **Run** on each cell. The notebooks are designed to execute step-by-step for ease of use.

## Output Interpretation

### Visualizations
- **Rating Distribution**: `rating_distributions.png`
- **Recommendation Probability**: `like_probabilities.png`
- **Bandit Algorithm Comparison**: `bandit_results.png`

### Metrics
- **Cumulative Reward**: Total liked recommendations over time.
- **Regret**: Difference between optimal and observed rewards.
- **Percentage Liked**: Fraction of recommendations liked by users.

## Experiments and Parameters
- **A/B Testing**: Testing periods of 1,000 and 5,000 visits.
- **Epsilon-Greedy**: Exploration rates (ε) of 0.05 and 0.10.
- **Softmax**: Temperature parameter τ = 0.1.
- **UCB**: Exploration parameter c = 1.0.