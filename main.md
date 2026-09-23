# Self-Attention: Understanding Q, K, V

## 1. Input Data

The input to the attention mechanism is the output of the **embedding layer**, with shape:

```
(B, T, C)
```

- **B** = Batch size (number of sequences processed in parallel)
- **T** = Tokens (sequence length)
- **C** = Embedding channels (embedding dimension)

A useful mental model:
- Each **row** = one token
- Each **column** = one dimension of that token's embedding

At this stage, every token has a vector describing its features, but tokens don't yet "know" anything about each other. Attention is the mechanism that lets them exchange information.

## 2. The Q, K, V Matrices

Once we have embeddings carrying per-token features, we need tokens to communicate with each other to understand relationships and context. To do this, we create three matrices:

| Matrix | Role | Intuition |
|---|---|---|
| **Q** (Query) | "What am I looking for?" | Each token asks questions |
| **K** (Key) | "What do I represent?" | Each token advertises what it offers |
| **V** (Value) | The refined content of each token | A cleaned-up, useful version of the embedding |

**Why do we need V if we already have the embedding?**
The raw embedding may contain noise or irrelevant information picked up during training. The V matrix acts as a filter/refinement step, keeping only the information that's actually useful for the task, similar to a learned projection that improves the representation.

These three matrices are obtained by multiplying the input by learned weight matrices `W_Q`, `W_K`, `W_V`:

```
(B, T, C) @ (C, D) = (B, T, D)
```

### Breaking down each matrix

**Q — Query matrix**
- Rows = tokens
- Columns = "questions" being asked
- Every token asks the *same set* of questions (column 1 = question A, column 2 = question B, etc.), but each token's *answer profile* differs based on its own content.

**K — Key matrix**
- Rows = tokens
- Columns = answers corresponding to Q's questions
- Column 1 of K answers the question in column 1 of Q, column 2 of K answers the question in column 2 of Q, and so on.

**V — Value matrix**
- Rows = tokens
- Columns = the actual "value"/content associated with each question
- Unlike K, V doesn't just say *whether* something matches a question — it carries the substantive information itself.

**Example to build intuition:**
- Q1 (column 1 of Q) = "Are you a verb?"
- K1 (column 1 of K) = the answer (yes / no / partially / similar, etc.)
- V1 (column 1 of V) = the actual content that represents "verb-ness"

**Why not just reuse K as V?**
Because K only answers *whether* something matches a specific question — it doesn't carry meaning on its own. Knowing "this is not a verb" doesn't tell you what the token actually *is* (noun? adjective?). V is what supplies that substantive information, regardless of the yes/no answer from K.

## 3. How Q, K, V Communicate: the Dot Product

Now that we have three matrices, how do they interact to produce something meaningful?

We use the **dot product**.

- Each row of Q = a token, columns = its questions
- Each row of K = a token, columns = its answers

By **transposing K**, we flip it so rows become questions and columns become tokens. Now we can compute a dot product between Q and Kᵀ.

For a single pair of tokens (Token 1 attending to Token 1):

```
(Q1_T1 * K1_T1) + (Q2_T1 * K2_T1) + (Q3_T1 * K3_T1) + ... + (Qn_T1 * Kn_T1) = Attention(T1, T1)
```

This gives a single number: *how much does Token 1's query match Token 1's key?* (i.e., how much attention Token 1 pays to itself).

Doing this for **every pair of tokens** produces a matrix of shape `(T, T)`:

- Each row = a "querying" token
- Each column = a "key" token
- Each value = how strongly the row-token's question is answered by the column-token

### From raw scores to attention weights

These raw dot-product values aren't very usable yet. We apply **softmax** (usually after scaling by `1/sqrt(D)`) to turn them into normalized weights between 0 and 1 — these are the **attention scores**. They tell us, for each token, how much "attention" it should pay to every other token (including itself).

### Weighted sum with V

Finally, we combine the attention scores with the **V** matrix:

```
scores (T, T) @ V (T, D) = output (T, D)
```

For a specific dimension, this looks like:

```
(score_T1_T1 * V1_T1) + (score_T1_T2 * V1_T2) + (score_T1_T3 * V1_T3) = new V1 for Token 1
```

In other words, the new representation of Token 1 is a **weighted blend** of the V vectors from all tokens, weighted by how relevant each token is to Token 1 — for example, 70% from itself, 20% from Token 2, 10% from Token 3.

This is the core idea of self-attention: **each token's new representation is a context-aware mixture of every other token's value**, weighted by relevance.

## 4. Masking

In some attention variants (like the decoder in autoregressive language models), tokens must **not** be allowed to look at future tokens — only at themselves and previous ones. This is enforced with a **mask**.

- **Unmasked (bidirectional) attention**: every token can attend to every other token, past and future. Useful for encoders (e.g., BERT-style models) where full-context understanding is needed.
- **Masked (causal) attention**: before applying softmax, we set the scores for "future" tokens to `-infinity` (or a very large negative number). After softmax, these become effectively 0, so a token can only attend to itself and earlier tokens.

This is what makes autoregressive generation possible: at inference time, the model predicts the next token using only the tokens it has already seen, and the mask enforces that same constraint during training.

## 5. Multi-Head Attention (MHA)

### Why one attention head isn't enough

A single softmax over attention scores forces the model to commit to **one** distribution of "relevance" per token. But language relationships are diverse — a token might need to attend to *syntax* (e.g., "which word does this verb agree with?") and *semantics* (e.g., "which word is this pronoun referring to?") at the same time. A single softmax can't represent multiple, independent types of relationships simultaneously — it would blend them into one averaged pattern.

**Solution: run multiple attention "heads" in parallel**, each with its own Q, K, V projections, allowing each head to specialize in a different kind of relationship (syntax, coreference, position, etc.).

### Shape transformation

Starting from:
```
(B, T, D)
```

We split the embedding dimension `D` into `C` heads, each with its own smaller dimension `d = D / C`:

```
(B, T, D) → (B, C, T, d)
```

- **B** = batch
- **C** = number of heads
- **T** = tokens
- **d** = dimension per head

Each head independently computes its own attention (its own Q, K, V, scores, and weighted V output) over shape `(B, T, d)`.

### Merging the heads back

After each head produces its own output, we concatenate all heads back together:

```
(B, C, T, d) → (B, T, C*d) = (B, T, D)
```

This restores the original shape. However, simply concatenating outputs from independent heads isn't enough — the model needs a way to let the heads' information **interact and combine** meaningfully.

### The Output matrix (W_O)

A final learned projection matrix, `W_O`, is applied to the concatenated output:

```
(B, T, D) @ (D, D) = (B, T, D)
```

This step mixes information across heads, allowing the model to combine the different "perspectives" (syntax, semantics, etc.) captured by each head into a single unified representation before passing it to the next layer.