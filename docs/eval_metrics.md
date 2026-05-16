# Evaluation Framework

The challenge employs a **multi-dimensional evaluation framework** to assess both *what you recommend* and *how you recommend it*.

---

## Composite Score

$$
\text{Score} = 0.50 \times \text{nDCG@20} + 0.10 \times \text{CatalogDiversity} + 0.10 \times \text{LexicalDiversity} + 0.30 \times \text{LLM-Judge}
$$

---

| Dimension | Weight | What it measures | How it is computed | Role in evaluation |
|---|---|---|---|---|
| **nDCG@20** | 0.50 | Ranking quality of the recommended tracks. | Computed from the ranked list of predicted tracks against the ground-truth relevant item. Higher-ranked correct recommendations receive more credit. | Primary recommendation metric. |
| **Catalog Diversity** | 0.10 | How broadly a system covers the music catalog. | Number of unique recommended tracks across all predictions divided by the total catalog size. | Complementary diversity indicator. |
| **Lexical Diversity** | 0.10 | How varied the generated language is. | Measured with Distinct-2, i.e., unique bigrams divided by total bigrams across generated responses. | Complementary response-generation indicator. |
| **LLM-as-a-Judge** | 0.30 | Quality of the generated explanation. | Blind-set responses are judged by a Gemini model used as an automatic judge. The judge evaluates two text-only dimensions: Personalization and Explanation Quality. These dimensions evaluate the written response independently from recommendation accuracy. To preserve the integrity of the blind evaluation, the organizers disclose the judge family but do not publish the evaluation prompt. | Blind-set response-quality evaluation. |

---

## Metric Definitions

### nDCG@k (Normalized Discounted Cumulative Gain)

Measures ranking quality by comparing the predicted ranking to the ideal ranking.

$$
\text{nDCG@k} = \frac{\text{DCG@k}}{\text{IDCG@k}}, \qquad \text{DCG@k} = \sum_{i=1}^{k} \frac{\mathbf{1}[\text{pred}_i \in \text{gold}]}{\log_2(i+1)}
$$

Each conversation turn has exactly **one** ground-truth track. The evaluation reports nDCG@1, nDCG@10, and nDCG@20. The primary metric is **nDCG@20**. Results are macro-averaged across sessions and turns.

---

### Catalog Diversity

$$
\text{CatalogDiversity} = \frac{|\text{unique recommended tracks}|}{|\text{catalog}|}
$$

Measures whether the system explores the full music catalog or concentrates on popular items.

---

### Lexical Diversity (Distinct-2)

$$
\text{LexicalDiversity} = \frac{|\text{unique bigrams across all responses}|}{|\text{total bigrams across all responses}|}
$$

Encourages varied and non-repetitive natural language generation.

---

### LLM-as-a-Judge

A large language model evaluates a **sampled subset of sessions** across multiple dimensions, each scored on a 1–5 integer scale. The detailed evaluation criteria are not publicly disclosed.

Before computing the composite score, each dimension is normalized to $[0, 1]$ using min-max normalization with fixed bounds ($\min=1$, $\max=5$):

$$
\text{score}_{\text{norm}} = \frac{\text{score} - 1}{5 - 1} = \frac{\text{score} - 1}{4}
$$

The normalized scores are then averaged and weighted at 0.30 in the composite.
