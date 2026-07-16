# ScoutDLRM: hybrid recommendation model for scouting events

An advanced, hybrid recommendation model for scouting events that matches scouts with optimal events based on deep semantic analysis of their interests, demographic criteria and categorical preferences, which integrates collaborative filtering, content-based and demographic approaches. ScoutDLRM is an adaptation of Facebook Deep Learning Recommendation Model (DLRM) architecture, improved with special text block fot text data analysis and ordinal regression idea for prediction of rates for each event by certain scout.

## Key Strengths & Technical Advantages

* **Specialized text block (Text MLP):** Unlike standard DLRM implementations that group all dense inputs together, this architecture isolates textual descriptions (user interests and event descriptions) into a specialized neural network block to perform deep semantic analysis.
* **State-of-the-Art embedding engine (Gemma 300M):** Using the `google/embeddinggemma-300m` sentence-transformer model to encode text data. It maximizes efficiency by using Matryoshka Representation Learning (MRL) to truncate embeddings down to 128 dimensions with minimal semantic loss.
* **Ordinal Regression idea:** Instead of treating user feedback as independent categories or arbitrary continuous values, the model solves an ordinal regression problem. It predicts a probability distribution over the 5-star rating scale using a Softmax layer and computes a weighted mean score to preserve structural ranking distances.
* **Advanced feature engineering:** Automatically handles cyclic features via a custom sine-cosine cyclic encoder and performs explicit temporal duration feature extraction out of date windows.
* **Gradio UI:** Shipped with a highly optimized Gradio web interface featuring custom CSS card components, an automated age-to-category mapping system, and real-time inference handling.

---

## Model Architecture

### 1. Preprocessing & Feature Engineering Pipeline
The system ingests multi-modal datasets and splits them into three independent channels before tensor formation:
* **Numeric features:** `age`, `price` (log-normalized), `min_age`, `max_age`, and `duration` (derived as `end_date - start_date + 1`).
* **Categorical features:** `category` (mapped dynamically across scout ranks), `type`, `currency`, `is_online`, and `season` (extracted cyclically).
* **Text features:** `interests` (scout profile hobbies) and `description` (event payload text).

```text
[Raw Input Fields]
│
├──► Numeric Fields ─────► Log Transform & Scaler ──────► [Bottom MLP] ─────┐
│                                                                           │
├──► Text Fields ────────► Gemma-300M Embedder (MRL) ───► [Text MLP] ───────┼─► [Interaction Layer] ─► [Top MLP] ─► [Softmax Head]
│                                                                           │
└──► Categorical Fields ─► Integer Label Mapping ───────► [Embed Bags] ─────┘
```

### 2. Interaction Layer Mechanics
The interaction layer captures explicit cross-feature correlations. It collects the continuous latent vectors output by the Bottom MLP, the text vectors from the Text MLP, and the sparse embeddings generated via `nn.EmbeddingBag` layers. It computes a batch dot-product matrix across all vector combinations and flattens only the unique upper-triangular elements to prevent feature redundancy.

$$\mathbf{A}_{\text{all}} = \left[ \mathbf{v}_{\text{dense}}, \mathbf{v}_{\text{text}}, \mathbf{v}_{\text{sparse},1}, \dots, \mathbf{v}_{\text{sparse},M} \right]^T \in \mathbb{R}^{(2+M) \times D}$$

$$\mathbf{X}_{\text{dot}} = \text{upper\\_tri\\_flat}(\mathbf{A}_{\text{all}} \times \mathbf{A}_{\text{all}}^T)$$

$$\mathbf{X}_{\text{interaction}} = \left[ \mathbf{v}_{\text{dense}} , \mathbf{X}_{\text{dot}} \right]$$

### 3. Ordinal Regression Head
To handle the discrete 5-star ranking structure correctly, the model uses a softened expectation layer instead of standard cross-entropy or mean squared error:

$$\mathbf{z} = \text{Top MLP}(\mathbf{X}_{interaction}) \in \mathbb{R}^5$$

$$P(\text{rating} = k) = \frac{\exp(z_k / T)}{\sum_{j=1}^5 \exp(z_j / T)}, \quad T = 0.2$$

$$\hat{y} = \sum_{k=1}^5 P(\text{rating} = k) \cdot k, \quad \text{where } k \in \{1, 2, 3, 4, 5\}$$

---

## Model Training & Evaluation

### Training Configuration
The ScoutDLRM model was trained on the processed multi-modal dataset using a combined optimization strategy to balance ordinal regression and categorical probability constraints.

| Parameter | Value |
| :--- | :--- |
| Optimizer | Adam |
| Learning Rate | 0.001 |
| Batch Size | 32 |
| Loss Function | Hybrid ($L_{\text{MSE}} + L_{\text{CE}}$) |
| Training Epochs | 18 |
| Softmax Temperature ($T$) | 0.2 |

### Performance Metrics on Test Dataset
The following table outlines the model's prediction accuracy and classification alignment evaluated on an independent test set:

| Metric | Value | 
| :--- | :--- | 
| Soft Agreement ($\epsilon = 0.25$) | 91% | 
| MAE | 0.1953 | 
| $R^2$ | 0.8867 | 
| Accuracy | 94% | 
| Macro F1-Score | 0.89 | 

---

## Repository File Structure

```text
.
├── dlrm_model.py                 # PyTorch ScoutDLRM neural network architecture
├── preprocessing_dlrm.py         # Feature engineering, CyclicEncoder, and Gemma transformer embedder
├── scout_dataset.py              # PyTorch custom Dataset and sparse-offset collate function
├── scout_app.py                  # Main Gradio application
└── requirements.txt              # Python environment package dependencies
```

---

## Installation

### Prerequisites
* Python 3.10+
* CUDA-compatible GPU (Optional, recommended for high-throughput embedding generation)

### Setup Environment
1. Clone the repository:
   ```bash
   git clone [https://github.com/Bulatova28/Hybrid-recommendation-model-for-scouting-events.git](https://github.com/Bulatova28/Hybrid-recommendation-model-for-scouting-events.git)
   cd scoutify-dlrm

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

## Usage

### Launching the Gradio Web Application

To run the production-ready interactive user interface locally, execute the main application script:
```bash
python scout_app.py
```
    
Once initialized, the interface will be available at http://localhost. The app contains a built-in pre-warming mechanism for the sentence-transformer model to ensure zero-lag execution upon the first user request.

