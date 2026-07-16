# Scoutify: Scout event recommendation system based on ScoutDLRM

An advanced, hybrid personalization platform designed for the National Organization of Scouts of Ukraine to match scouts with optimal events based on demographic criteria, categorical preferences, and deep semantic text profiling. The core recommendation pipeline is powered by **ScoutDLRM**, a tailored adaptation of Facebook Deep Learning Recommendation Model architecture, featuring low-dimensional transformer text embeddings and an expectation-based ordinal regression head.

## Key Strengths & Technical Advantages

* **Specialized Deep Text Profiling (Text MLP):** Unlike standard DLRM implementations that group all dense inputs together, this architecture isolates textual descriptions (user interests and event descriptions) into a specialized neural network block to extract deep semantic features.
* **State-of-the-Art Embedding Engine (Gemma 300M):** Leverages the `google/embeddinggemma-300m` sentence-transformer model to encode text data. It maximizes efficiency by using Matryoshka Representation Learning (MRL) to truncate embeddings down to 128 dimensions with minimal semantic loss.
* **Expectation-Based Ordinal Regression Head:** Instead of treating user feedback as independent categories or arbitrary continuous values, the model solves an ordinal regression problem. It predicts a probability distribution over the 5-star rating scale using a Softmax layer and computes a softened weighted mathematical expectation score to preserve structural ranking distances.
* **Advanced Feature Engineering:** Automatically handles cyclical periodic variations via a custom sine-cosine cyclic encoder and performs explicit temporal duration feature extraction out of date windows.
* **Production-Ready UI Environment:** Shipped with a highly optimized Gradio web interface featuring custom CSS card components, an automated age-to-category mapping system, and real-time inference handling.

---

## System Architecture & Core Innovations

### 1. Preprocessing & Feature Engineering Pipeline
The system ingests multi-modal datasets and splits them into three independent channels before tensor formation:
* **Continuous Features:** `age`, `price` (log-normalized), `min_age`, `max_age`, and `duration` (derived as `end_date - start_date + 1`).
* **Categorical Features:** `category` (mapped dynamically across scout ranks), `type`, `currency`, `is_online`, and `season` (extracted cyclically).
* **Textual Features:** `interests` (scout profile hobbies) and `description` (event payload text).

[Raw Input Fields]
│
├──► Continuous Fields ──► Log Transform & Scaler ──────► [Bottom MLP] ──┐
│                                                                        │
├──► Text Fields ────────► Gemma-300M Embedder (MRL) ──► [Text MLP] ─────┼─► [Interaction Layer] ─► [Top MLP] ─► [Softmax Expectation Head]
│                                                                        │
└──► Categorical Fields ─► Integer Label Mapping ───────► [Embed Bags] ──┘

### 2. Interaction Layer Mechanics
The interaction layer captures explicit cross-feature correlations. It collects the continuous latent vectors output by the Bottom MLP, the text vectors from the Text MLP, and the sparse embeddings generated via `nn.EmbeddingBag` layers. It computes a batch dot-product matrix across all vector combinations and flattens only the unique upper-triangular elements to prevent feature redundancy.

$$\mathbf{A}_{\text{all}} = \left[ \mathbf{v}_{\text{dense}}, \mathbf{v}_{\text{text}}, \mathbf{v}_{\text{sparse},1}, \dots, \mathbf{v}_{\text{sparse},M} \right]^T \in \mathbb{R}^{(2+M) \times D}$$

$$\mathbf{X}_{\text{dot}} = \text{upper\tri\flat}(\mathbf{A}_{\text{all}} \mathbf{A}_{\text{all}}^T)$$

$$\mathbf{X}_{\text{interaction}} = \left[ \mathbf{v}_{\text{dense}} \,\Vert{}\, \mathbf{X}_{\text{dot}} \right]$$

### 3. Ordinal Regression Head
To handle the discrete 5-star ranking structure correctly, the model uses a softened expectation layer instead of standard cross-entropy or mean squared error:

$$\mathbf{z} = \text{Top\_MLP}(\mathbf{X}_{interaction}) \in \mathbb{R}^5$$

$$P(\text{rating} = k) = \frac{\exp(z_k / T)}{\sum_{j=1}^5 \exp(z_j / T)}, \quad T = 0.2$$

$$\hat{y} = \sum_{k=1}^5 P(\text{rating} = k) \cdot k, \quad \text{where } k \in \{1, 2, 3, 4, 5\}$$

---

## Repository File Structure
.
├── dlrm_model.py                 # PyTorch ScoutDLRM neural network architecture
├── preprocessing_dlrm.py         # Feature engineering, CyclicEncoder, and Gemma transformer embedder
├── scout_dataset.py              # PyTorch custom Dataset and sparse-offset collate function
├── scout_app.py                  # Main Gradio application, authentication UI, and inference wrapper
├── event_df.pkl                  # Serialized event catalog pandas DataFrame
├── feature_transformer_gemma.pkl # Saved data preprocessing pipeline state
├── scout_dlrm_dyploma_gemma.pth  # Trained model checkpoint parameters
└── requirements.txt              # Python environment package dependencies

---

## Installation

### Prerequisites
* Python 3.10+
* CUDA-compatible GPU (Optional, recommended for high-throughput embedding generation)

### Setup Environment
1. Clone the repository:
   ```bash
   git clone [https://github.com/yourusername/scoutify-dlrm.git](https://github.com/yourusername/scoutify-dlrm.git)
   cd scoutify-dlrm

2. Install dependencies:
    ```bash
    pip install -r requirements.txt

3. Ensure the following files are present in the root directory before launching:
- scout_dlrm_dyploma_gemma.pth
- feature_transformer_gemma.pkl
- event_df.pkl

## Usage

### 1. Launching the Gradio Web Application
To run the production-ready interactive user interface locally, execute the main application script:
    ```bash
    python scout_app.py

Once initialized, the interface will be available at http://localhost. The app contains a built-in pre-warming mechanism for the sentence-transformer model to ensure zero-lag execution upon the first user request.

## Model Hyperparameters
The neural network configurations hardcoded into the system execution layers are mapped as follows: 
- **Embedding Framework**: 3 distinct tracking tables with an embedding vector size of 32 (embed_dim=32) processing multi-categorical scout structures. 
- **Bottom MLP Dimension Layers**: [8 -> 128 -> 64 -> 32] for continuous dense parameter mapping with a internal dropout regularization factor of 0.2. 
- **Text MLP Dimension Layers**: [256 -> 64 -> 32] specialized text tensor processor tracking the truncated 128-dimensional multi-token Gemma embeddings. 
- **Interaction Input Vector Dimension**: 42 total cross-features combined with continuous indices[cite: 4].Top MLP Dimension Layers: [42 -> 16 -> 5] predicting categorical scores passed straight into the Softmax Expectation Head layer. 