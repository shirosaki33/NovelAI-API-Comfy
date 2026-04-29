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
- `NovelAI Retry Settings`

## Reference and editing nodes

- `NovelAI 💎 Precise Reference`
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

Use `NovelAI Character (V4.5)` for one character block and combine multiple blocks with `NovelAI Character Stack (V4.5)` (up to 8 inputs).

## Notes

- Includes retry handling through the separate `NovelAI Retry Settings` node.
- T2I/I2I output Anlas balance, last generation cost, total tracked cost, and status text after generation.
- `NovelAI Anlas` also works as a manual tracker/check node using the same internal cost counter.
- Includes the extra samplers `k_dpm_2` and `k_dpm_fast`.
- Precise Reference is V4.5-only and reference images are resized/padded before being sent to the API.
- Director tool nodes are included on a best-effort basis and may need payload fine-tuning depending on API-side changes.
