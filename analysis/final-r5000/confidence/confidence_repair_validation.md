# Verbalized-confidence repair — validation
**Rule:** strip one trailing <|im_end|> sentinel + optional <..> wrap; accept int in [0,100]
## Natural terminal confidence
- rows: 5,000
- status before: {'malformed': 4354, 'missing': 637, 'parsed': 9}
- status after:  {'recovered': 4352, 'missing': 639, 'parsed': 9}
- malformed recovered: 4,352 / 4,354 (not recovered: 2)
- previously-parsed values unchanged: True
- all final values integers in [0,100]: True
- non-recovered forms: {'100%<|im_end|>': 2}
- **evaluable natural answers:** 3,550; **usable confidence among them:** 3,542 (3,166 correct / 376 incorrect)
- recovered/final distribution: {'n': 4361, 'mean': 93.45, 'median': 95.0, 'min': 0, 'max': 100, 'pct_ge_90': 74.7, 'pct_eq_100': 36.5}
- examples recovered: [['100<|im_end|>', '100'], ['100<|im_end|>', '100'], ['100<|im_end|>', '100'], ['99<|im_end|>', '99'], ['100<|im_end|>', '100'], ['100<|im_end|>', '100']]
- examples NOT recovered: [['100%<|im_end|>'], ['100%<|im_end|>']]

## Checkpoint confidence
- rows: 55,000
- status before: {'malformed': 48497, 'parsed': 5252, 'missing': 1251}
- status after:  {'recovered': 47227, 'parsed': 5252, 'missing': 2521}
- malformed recovered: 47,227 / 48,497 (not recovered: 1,270)
- previously-parsed values unchanged: True
- all final values integers in [0,100]: True
- non-recovered forms (top): {'<integer 0-100><|im_end|>': 1053, '<integer 0-100>': 131, '': 35, '<integer 100><|im_end|>': 14, '<': 13, '<integer 0-100>"': 9, '50%<|im_end|>': 3, '<integer 90><|im_end|>': 2}
- final distribution: {'n': 52479, 'mean': 91.69, 'median': 95.0, 'min': 0, 'max': 100, 'pct_ge_90': 60.3, 'pct_eq_100': 28.8}

### Per-fraction coverage after repair
| fraction | n | parsed | recovered | usable | coverage % |
|---:|---:|---:|---:|---:|---:|
| 0.0 | 5,000 | 3,435 | 1,175 | 4,610 | 92.2 |
| 0.1 | 5,000 | 846 | 3,868 | 4,714 | 94.3 |
| 0.2 | 5,000 | 348 | 4,372 | 4,720 | 94.4 |
| 0.3 | 5,000 | 186 | 4,504 | 4,690 | 93.8 |
| 0.4 | 5,000 | 142 | 4,589 | 4,731 | 94.6 |
| 0.5 | 5,000 | 100 | 4,653 | 4,753 | 95.1 |
| 0.6 | 5,000 | 68 | 4,688 | 4,756 | 95.1 |
| 0.7 | 5,000 | 62 | 4,718 | 4,780 | 95.6 |
| 0.8 | 5,000 | 41 | 4,808 | 4,849 | 97.0 |
| 0.9 | 5,000 | 22 | 4,876 | 4,898 | 98.0 |
| 1.0 | 5,000 | 2 | 4,976 | 4,978 | 99.6 |
