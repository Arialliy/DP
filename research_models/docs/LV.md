# DeepPro-LV: level–variation decomposition of temporal profiles

## 1. Problem the model addresses

DeepPro/DeepPro-Plus read each pixel's **temporal profile** x_p(t) over a T = 40-frame window and detect a target as a *temporal anomaly*. Their difference convolutions annihilate any constant: the TD kernel `2w_t − w_{t+1} − w_{t−1}` maps a constant profile to 0. TPro is a linear mixing of time steps followed by 1×1×1 convs, and the spatial context is a 19×19 window on single noisy frames, *before* any temporal integration.

A target that does not move relative to its background therefore produces a flat profile, which this machinery treats as background. Measured on our data with the released checkpoint at 1 false alarm/frame:
- val Pd is 0.43 for stationary targets (relative speed < 0.2 px/frame) vs 0.92 for medium/fast ones;
- stationary targets are 76% of all misses;
- a controlled insertion probe gives detection rates of 0.03 / 0.75 / 0.80 at 0 / 0.5 / 1 px/frame for the same +20 grey-level target, and 0.00 vs 0.90 for a −40 (dark) target (`results/probe/official_rgbt.json`).

## 2. Observation: split each profile into level and variation

For a window X ∈ R^{T×H×W} (grey levels normalised with the authors' RGB-T mean/std):

```
L_p    = median_t X_p(t)                 level (persistent component), one image per window
V_p(t) = X_p(t) − L_p                     variation (transient component), T images
E_p    = mean_t |V_p(t)|                  variation energy (where something changed in the window)
```

**Dwell-time complementarity.** Take a target of contrast a that covers pixel p in k of the T frames, over a background whose level is otherwise b. Then:
- if k < T/2, L_p = b, so V_p(t) = a exactly during the k frames and 0 otherwise: the whole signal is transient;
- if k > T/2, L_p = b + a, so V_p(t) = −a during the T − k absent frames: the target sits in the level, and V only marks its arrival or departure.

So whatever the speed, the target's signal is in V, in L, or in both, never in neither. DeepPro's machinery is suited to V; L needs a *spatial* detector, because a level image of a stationary target looks like a single, noise-averaged frame. Padded (invalid) frames in a training window are excluded from the median (`torch.nanmedian`), and V is set to 0 there.

## 3. Architecture (`code/lv.py`, registry name `lv_full`)

| Block | Input → output | Details | Params |
|---|---|---|---|
| Decomposition | X (b,1,T,H,W) → L (b,1,H,W), V (b,1,T,H,W), E (b,1,H,W) | median over time; no parameters | 0 |
| **Variation pathway** | V → per-frame logit l_v (b,T,H,W) | Unmodified DeepPro-Plus: SD-conv (5×7×7, 1→8), 2 STD-ResBlocks (8→16→32), TPro (8 learnable 40×40 matrices), 1×1×1 conv 32→8 → 8→1 | 70,913 |
| **Level pathway** | [L, E] (b,2,H,W) → logit l_L (b,H,W) | 2-D encoder–decoder, 3 scales (16/32/64 ch, two 3×3 conv+BN+ReLU per stage, max-pool ↓, bilinear ↑ with skips), 1×1 head | 118,769 |
| **Consistency gate** | [V(t), \|V(t)\|] → g(t) ∈ (0,1) (b,T,H,W) | per-pixel MLP (1×1×1 convs 2→8→1, sigmoid); output bias initialised at +2 (open) | 33 |
| **Fusion** | l_v(t), l_L, g(t) → fused logit | noisy-OR: p(t) = 1 − (1 − σ(l_v(t))) · (1 − g(t)·σ(l_L)), computed in log space and returned as a logit | 0 |
| Total |  |  | **189,715** |

**Why the gate exists.** If a target dwells longer than half the window but not the whole window, L shows it at every frame. At the frames where it is absent, V(t) ≈ −a (the current pixel is darker than the level), and the gate learns to switch the level evidence off there.

**Why noisy-OR rather than a learned mixing layer.** The two pathways answer disjoint questions ("did something transient happen here now?" and "is there a persistent target here?"). Noisy-OR makes each pathway's contribution directly readable (`code/explain.py`), and it keeps the variation pathway exactly a DeepPro-Plus detector.

## 4. Training

Identical for all compared models (`code/train.py`); this is the authors' RGB-T recipe:
- 40-frame windows ending at every frame i ≥ 4; short windows are zero-padded at the front or back; clips are sampled ∝ window length.
- 128×128 crops, centred (±21 px jitter) on a random target of the middle frame with p = 0.75.
- Adam (lr 1e-3, weight decay 1e-4), lr × 0.7 every 10 epochs, 32 epochs, batch 4, sample rate 0.05 (175 iterations/epoch on our 32 training clips).

**Loss.** L = SoftIoU(fused logits, Gaussian masks) + λ · SoftIoU(l_L, P), with λ = 0.5. Here P is the persistent-target mask: pixels covered by a target mask in at least 75% of the window's valid frames. The term makes the level pathway specialise in persistent targets; models without a level pathway use the first term only.

**Checkpoint.** The final epoch for every model. The authors' released script picked the best epoch by test IoU; we do no checkpoint selection at all. Training runs on CPU (Xeon 8476C), 8 threads per run, fp32, seeds 0/1/2.

## 5. Inference

The authors' windowing: T = 40, stride 36, the last window aligned to the clip end, max over overlaps. The decomposition is recomputed in every window, so no state is kept across windows and no future beyond the window is used. Like DeepPro, the model is *offline within a window*: frame t uses up to 39 frames of the same window, including later frames. The output is a per-pixel, per-frame logit; detections are local maxima (9×9).

## 6. Controls built into the code

| Name | Change | Tests |
|---|---|---|
| `deeppro_plus` | baseline, input X | — |
| `lv_vonly` | variation pathway on V, no level pathway | Does background removal alone help or hurt? Expected: stationary targets are lost |
| `lv_xlevel` | variation pathway on X (unchanged DeepPro-Plus) + level pathway + gate | Is V-input needed, or only the extra level pathway? |
| `lv_frame` | as `lv_full`, but the 2-D net sees each *current frame* X(t) (+E) instead of L, run per frame (same parameters, aux target) | Is it the *level* observation, or "any spatial branch"? |
| `lv_nogate` | `lv_full` without the gate | Value of the consistency gate |
| `dpp_wide` | DeepPro-Plus with widened channels (12/32/56), 198,457 params | Are gains just from parameters? |

All controls use the same 40 frames, schedule, crops and loss, so frames, training length and parameter count are held fixed or matched across the comparison.
