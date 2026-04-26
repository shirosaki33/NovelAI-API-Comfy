# ComfyUI NovelAI

Standalone ComfyUI nodes for the NovelAI image API.

## Install

Copy the folder `comfyui_novelai` to:

```txt
ComfyUI/custom_nodes/comfyui_novelai
```

Restart ComfyUI.

## Token priority

1. `ComfyUI/custom_nodes/comfyui_novelai/token.txt`
2. `ComfyUI/.env` with `NAI_ACCESS_TOKEN=...`
3. Environment variables
4. `api_token` input connection

## Token connector node

Use:

- `NovelAI Token`

It outputs a `STRING` token connector so you only type the token once and connect it to every node that has an `api_token` input.

## Core nodes

- `NovelAI Token`
- `NovelAI Parameters`
- `NovelAI Character`
- `NovelAI Character Stack`
- `NovelAI T2I`
- `NovelAI I2I`
- `NovelAI Anlas`

## Reference and editing nodes

- `NovelAI 💎 Precise Reference`
- `NovelAI 💎 Vibe Transfer`
- `NovelAI 💎 Inpaint`
- `NovelAI 💎 Enhance`
- `NovelAI 💎 Upscale`

## Director tool nodes

- `NovelAI 💎 Remove Background`
- `NovelAI 💎 Line Art`
- `NovelAI 💎 Sketch`
- `NovelAI 💎 Colorize`
- `NovelAI 💎 Emotion`
- `NovelAI 💎 Declutter`

Nodes marked with `💎` may spend Anlas.

## Character system

Use `NovelAI Character` for one character block and combine multiple blocks with `NovelAI Character Stack` (up to 20 inputs).

## Notes

- Includes retry handling for transient API/network errors.
- Includes Anlas reporting (`anlas_text`, `anlas_before`, `anlas_after`, `actual_cost`).
- Includes the extra samplers `k_dpm_2` and `k_dpm_fast`.
- Director tool nodes are included on a best-effort basis and may need payload fine-tuning depending on API-side changes.
