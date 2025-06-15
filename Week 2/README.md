
# StoryGAN: A Sequential Conditional GAN for Story Visualization

## 1. Model Architecture

**StoryGAN** is a sequential conditional GAN model that generates a coherent sequence of images based on a multi-sentence story. The architecture ensures both local (sentence-level) and global (story-level) consistency.

### Components

- **Story Encoder**: Encodes the entire story `S` into a latent vector `h_0` using a stochastic process:

  ```
  h_0 = mu(S) + sigma(S) * epsilon,  where epsilon ~ N(0, I)
  ```

- **Context Encoder**: A deep RNN composed of:
  - **GRU Layer**: Processes the current sentence `s_t` and random noise `epsilon_t` to produce `i_t`.
  - **Text2Gist Cell**: Combines `i_t` and `h_(t-1)` to output a "gist" vector `o_t`, using dynamic filtering:

    ```
    o_t = Filter(i_t) * h_t
    ```

- **Image Generator**: Takes `o_t` and produces an image `x_hat_t`.

- **Discriminators**:
  - **Image Discriminator (D_I)**: Checks whether an image matches the sentence and initial context.
  - **Story Discriminator (D_S)**: Assesses whether the full image sequence matches the story.

    ```
    D_S = sigmoid(w^T (E_img(X) * E_text(S)) + b)
    ```

---

## 2. Network Architecture (Layer-wise)

### 2.1 Story Encoder

| Layer | Operation                  | Input Shape    | Output Shape |
|-------|----------------------------|----------------|---------------|
| 1     | Linear + BN + ReLU         | 128 × T        | 128           |
| 2     | Gaussian Sampling           | 128            | 128           |

---

### 2.2 Context Encoder

**Input:** Sentence vector (128) + noise vector (e.g., 100)

| Layer | Operation                  | Input Shape          | Output Shape |
|-------|----------------------------|----------------------|---------------|
| 1     | Linear + BN + ReLU         | 128 + noise_dim      | 128           |
| 2     | GRU                        | 128                  | 128           |
| 3     | Text2Gist Cell             | (i_t: 128, h_prev: 128) | o_t: 128    |

---

### 2.3 Filter Network (used inside Text2Gist)

| Layer | Operation            | Input | Output Shape         |
|-------|----------------------|-------|----------------------|
| 1     | Linear + BN + Tanh   | 128   | 1024                 |
| 2     | Reshape              | 1024  | [16, 1, 1, 64]       |

---

### 2.4 Image Generator

**Input:** Gist vector `o_t`

| Layer | Operation                        | Output Shape     | Notes              |
|-------|----------------------------------|------------------|--------------------|
| 1     | Conv2D (3x3, 512) + BN + ReLU    | [512, 4, 4]      |                    |
| 2     | Upsample ×2                      | [512, 8, 8]      |                    |
| 3     | Conv2D (3x3, 256) + BN + ReLU    | [256, 8, 8]      |                    |
| 4     | Upsample ×2                      | [256, 16, 16]    |                    |
| 5     | Conv2D (3x3, 128) + BN + ReLU    | [128, 16, 16]    |                    |
| 6     | Upsample ×2                      | [128, 32, 32]    |                    |
| 7     | Conv2D (3x3, 64) + BN + ReLU     | [64, 32, 32]     |                    |
| 8     | Upsample ×2                      | [64, 64, 64]     |                    |
| 9     | Conv2D (3x3, 3) + Tanh           | [3, 64, 64]      | Final RGB Image    |

---

### 2.5 Image Discriminator

**Input:** Image `x_t`, sentence `s_t`, and context `h_0`

| Layer | Operation                        | Output Shape     | Notes                       |
|-------|----------------------------------|------------------|-----------------------------|
| 1     | Conv2D (4x4, 64) + BN + LeakyReLU| [64, 32, 32]     |                             |
| 2     | Conv2D (4x4, 128) + BN + LeakyReLU| [128, 16, 16]   |                             |
| 3     | Conv2D (4x4, 256) + BN + LeakyReLU| [256, 8, 8]     |                             |
| 4     | Conv2D (4x4, 512) + BN + LeakyReLU| [512, 4, 4]     |                             |
| 5     | Conv2D (3x3, 512) + BN + LeakyReLU| [512, 4, 4]     | Combines conditional input  |
| 6     | Conv2D (4x4, 1) + Sigmoid        | [1, 1, 1]        | Final probability score     |

---

### 2.6 Story Discriminator

#### Image Encoder

| Layer | Operation                   | Output Shape        |
|-------|-----------------------------|---------------------|
| 1-4   | Conv2D + BN + LeakyReLU     | [512, H, W]         |
| 5     | Conv2D (4x4, 32) + BN       | [32, 1, 1]          |
| 6     | Reshape + concat over T     | [1, 32 × 4 × T]     |

#### Text Encoder

| Layer | Operation         | Input Shape   | Output Shape       |
|-------|-------------------|---------------|---------------------|
| 1     | Linear + BN       | 128 × T       | 32 × 4 × T          |

#### Final Scoring:

```
D_S = sigmoid(w^T * (E_img(X) * E_text(S)) + b)
```

---

## 3. Output

For a story of `T` sentences, the model generates a sequence of images:

```
X_hat = [x̂_1, x̂_2, ..., x̂_T], where each x̂_t is in R^{3x64x64}
```

---

## 4. Loss Functions

The total loss function for training StoryGAN is:

```
min_θ max_ψI,ψS   α * L_image + β * L_story + L_KL
```

### Components:

- **KL Divergence**:
  ```
  L_KL = KL(N(mu(S), diag(sigma^2(S))) || N(0, I))
  ```

- **Image Discriminator Loss**:
  ```
  L_image = sum over t [log D_I(x_t, s_t, h_0) + log(1 - D_I(x̂_t, s_t, h_0))]
  ```

- **Story Discriminator Loss**:
  ```
  L_story = log D_S(X, S) + log(1 - D_S(X̂, S))
  ```

---

## 5. Training Procedure

**Inputs:**
- A story `S = [s_1, s_2, ..., s_T]`
- Ground-truth images `X = [x_1, x_2, ..., x_T]`

**Steps:**

1. Encode each sentence `s_t` to a 128-dim vector.
2. Story encoder maps full `S` to latent `h_0`.
3. For each time step `t`:
   ```
   i_t = GRU(s_t, epsilon_t)
   o_t = Text2Gist(i_t, h_{t-1})
   x̂_t = Generator(o_t)
   ```
4. Update discriminators:
   - `D_I` with real/fake sentence-image pairs.
   - `D_S` with real/fake full sequences.
5. Use Adam optimizer and alternate generator/discriminator updates.

---

## 6. Key Innovations

- **Text2Gist**: Combines dynamic local sentence info and global context with filter-based gating.
- **Two-level Discriminators**: Improve both per-frame quality and story coherence.
- **Stochastic Story Encoder**: Adds diversity and robustness.
- **Training Strategy**: Alternating discriminator and generator updates ensures balanced learning.
