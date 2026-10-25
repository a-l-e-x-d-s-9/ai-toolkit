# MOSS training-only LoRA adapters

MOSS supports `model.assistant_lora_path` and a signed
`model.assistant_lora_strength` (default `1.0`). The UI exposes these as
**Training Adapter Path** and **Training Adapter Strength** for MOSS variants.
The strength setting currently applies to MOSS only; other architectures reject
a non-default value instead of silently ignoring it.

To train a new LoRA or LoKr on a checkpoint containing a known turbo merge:

```yaml
model:
  arch: moss_image
  name_or_path: /path/to/my_turbo_merged_checkpoint.safetensors
  assistant_lora_path: /path/to/the_exact_turbo_adapter.safetensors
  assistant_lora_strength: -1.0
```

This is a fragment to add to your existing training configuration. Keep your
normal MOSS training scheduler, timestep distribution and other settings.
These options do not select a different training objective or set sample steps.

If the checkpoint contains `W + a * turbo`, use strength `-a`. Training uses
`checkpoint - a * turbo`; previews apply the inverse, using the loaded checkpoint
plus the newly trained network. For the MOSS SFT 8-step DMD adapter, configure
previews with `sample.sample_steps: 8`, `sample.guidance_scale: 1`, and the normal
MOSS flowmatch sampler with shift 5. The adapter version and original merge
strength must match. Track them with each checkpoint.

For example, when the original merge strength was 0.8, set
`assistant_lora_strength: -0.8`. Positive strength adds a training-only adapter;
zero disables loading it, including during previews.

The assistant is frozen and separate from the new trainable network. For LoRA
or LoKr training, it is excluded from that network's optimizer and saved adapter.
At inference, load the newly trained adapter on the original turbo checkpoint
without an additional copy of the turbo adapter.

The model is adjusted once in memory before post-load quantization and memory
management. Already quantized checkpoints use the toolkit's dequantize/add/
requantize merge. Previews use an inverse runtime adapter; they do not repeatedly
requantize the base weights. The assistant stays inactive on CPU between previews.
On resume the adjustment is reapplied once to the freshly loaded checkpoint.
The checkpoint file on disk is never rewritten by this loading operation.

INT8 merges and subsequent training/merging can introduce rounding error, so
subtraction does not promise exact recovery of an earlier full-precision model.

Supported fixed adapters are standard linear LoRAs in dotted Comfy
(`diffusion_model.*`, `lora_down`/`lora_up`, optional alpha) or PEFT
(`transformer.*`, `lora_A`/`lora_B`) form. Independent Q/K/V projections are
fused exactly, preserving per-projection scale and per-module rank. Unsupported
keys and shape mismatches fail explicitly. LoKr remains supported as the newly
trained network, but not as this fixed assistant adapter. Combining the assistant
with `model.lora_path` or `model.inference_lora_path` is rejected.

CPU regression tests (no model download):

```bash
CUDA_VISIBLE_DEVICES="" HF_HUB_OFFLINE=1 python -m unittest testing.test_moss_training_adapter -v
```
