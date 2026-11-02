# Stage 2 — Does the LLM answer with the voice or the words?

## Setup

- **Question (one wording):** "How does the speaker sound in this recording? Answer with one word: angry, happy, neutral, or sad."
- **Answer:** the highest first-token log-probability among the four answer words (checked against generated text on a few clips).
- **Models:** Qwen2-Audio-7B-Instruct and Audio Flamingo 3 (LLM body in 8-bit on a 16 GB GPU), Voxtral-Mini-3B (bf16).

## Results

| Model | Data | Conflict → voice | → words | other |
|---|---|---|---|---|
| Qwen2-Audio | 240 held-out clips | **7.5%** | **80.8%** | 11.7% |
| Qwen2-Audio | pilot, 36 conflict clips | 1 | 35 | 0 |
| Audio Flamingo 3 | pilot, 36 conflict clips | 8 | 23 | 5 |
| Voxtral-Mini | pilot, 36 conflict clips | 0 | 36 | 0 |

More facts on Qwen2-Audio:

- **It is confident when wrong:** mean probability ≈ 0.90 when it follows the words.
  Example: an implicitly-angry sentence read in a happy StyleTTS2 voice → "angry", probability 0.99.
- **The voice barely changes the answer.** For the same sentence and TTS, all four voice versions get the same answer in 75% of groups.
- **When the voice is the only clue** (neutral words), it is right only 33% of the time (chance 25%).
- **Agreeing clips:** 72% correct; neutral-voice agreeing clips only 44% (it tends to say "angry" or "sad").
- **Prompting doesn't fix it:** "Ignore what the words mean. Judging only by the voice, how does the speaker feel?" → 6.9% voice on conflict clips.

## What it means

Combined with Stage 1, this is a **usage failure**: the encoder's middle layers carry the voice at ~95%, but the LLM answers from the words. The bias is not a prompt-following issue.

## Not released yet

Per-layer probes inside the LLM (last-token and audio-token states). Only a 60-clip pilot is analysed so far, with probes trained on ~48 clips × 4,096 features, which is too small to read layer by layer. The full 1,248-clip run will be added here.
