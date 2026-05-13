# OpenAIParameter Golf Challenge: Recurrent + Bigram Transformer Experiment

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

## Why This Result Was Meaningful

Although this project did not reach leaderboard-level validation performance, the result was still meaningful because it successfully demonstrated several of the core goals of the Parameter Golf challenge under local hardware constraints.

The challenge emphasizes parameter efficiency, compressed artifact size, and creative architecture design rather than simply scaling model size. My best experiment achieved a compressed serialized artifact size of only `4.61 MB`, which is significantly below the challenge’s `16 MB` limit while still maintaining reasonable language modeling performance.

The project also demonstrated that recurrent/shared transformer depth can improve parameter efficiency. Instead of allocating a separate set of parameters for every transformer layer, the model reused only 4 unique transformer blocks across 8 logical layers. This allowed deeper effective computation while keeping parameter count and compressed size small.

In addition, I explored lightweight token-statistical calibration inspired by leaderboard approaches such as `Calib32 Token-Only N-gram + AsymLogit Stack`. I introduced learned unigram and bigram logit bias tensors into the transformer output pipeline, creating a hybrid neural/statistical language modeling approach.

Compared to the leaderboard methods, my implementation remained much closer to a conventional transformer architecture rather than a highly specialized token-only compression system. The leaderboard models appear to rely heavily on optimized token-statistical modeling and calibration pipelines specifically tuned for compression-oriented evaluation metrics such as `val_bpb`.

Despite running locally on Apple Silicon with partial validation (`VAL_MAX_BATCHES=100`) rather than full 8xH100 evaluation, the project still achieved:

- recurrent shared-block transformer implementation
- hybrid transformer + bigram calibration architecture
- compressed artifact under 16MB
- reproducible training experiments
- measurable improvement over baseline transformer configurations

The experiments established a strong foundation for future work involving:
- trigram or hashed n-gram calibration
- asymmetric logit calibration
- low-rank parameterization
- quantization-aware training
- more advanced compression-oriented architectures
- 
## Future Work

- Run full validation on CUDA/H100
- Port final MLX changes fully into CUDA `train_gpt.py`
- Add trigram or hashed n-gram bias
- Explore asymmetric logit calibration
- Test low-rank linear layers and quantization-aware training
