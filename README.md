# Parameter Golf: Recurrent + Bigram Transformer Experiment

This repository documents a non-record local Apple Silicon MLX experiment inspired by OpenAI Parameter Golf. The goal was to explore small language model architectures that fit within the spirit of the 10-minute / 16MB challenge.

## Best Result

| Run | Architecture | Validation | val_bpb | Compressed Size |
|---|---|---|---:|---:|
| recur8_4_448_bigram | 8 logical layers / 4 unique shared blocks / dim 448 + unigram/bigram logit bias | partial local validation, VAL_MAX_BATCHES=100 | 2.0305 | 4.61 MB |

This is not an official leaderboard result because it was run locally on Apple Silicon with partial validation, not full validation on 8xH100.

## What I Did

I started from the baseline MLX training script and tested several transformer configurations. The strongest plain transformer baseline I found was a 6-layer, dim-448 model. I then implemented recurrent/shared transformer blocks using `NUM_UNIQUE_BLOCKS`, allowing the model to reuse fewer unique transformer blocks across more logical layers.

The best recurrent-only model used 8 logical layers and 4 unique shared blocks. This improved local training loss compared with the 6-layer baseline.

After that, I added a lightweight token-statistical calibration component inspired by token-only n-gram leaderboard methods. The model adds learned unigram and bigram logit biases to the transformer logits before softmax.

## Main Code Changes

- Added `NUM_UNIQUE_BLOCKS` for recurrent/shared transformer blocks
- Used modulo indexing to reuse blocks across logical depth
- Added learned unigram and bigram logit bias
- Added `VAL_MAX_BATCHES` for faster local Apple Silicon validation
- Verified compressed artifact size was below 16MB

## Comparison

| Experiment | Result |
|---|---|
| Baseline 6-layer dim-448 | train_loss 3.5362 at 700 steps |
| Recurrent 8 logical / 4 unique dim-448 | train_loss 3.5124 at 700 steps |
| Recurrent + bigram bias | val_bpb 2.0305, size 4.61 MB |

## Future Work

- Run full validation on CUDA/H100
- Port final MLX changes fully into CUDA `train_gpt.py`
- Add trigram or hashed n-gram bias
- Explore asymmetric logit calibration
- Test low-rank linear layers and quantization-aware training