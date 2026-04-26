ComfyUI nodes for full integration with the NovelAI Image Generation API.

Includes T2I, I2I, character prompt system, Precise Reference & Vibe Transfer, inpainting, enhance, upscale, and multiple director tools — with built-in retry handling and real-time Anlas tracking.

Supports flexible token usage:
- Place your API token in `ComfyUI/custom_nodes/comfyui_novelai/token.txt`
- Or use `.env` with `NAI_ACCESS_TOKEN=...`
- Or connect it through the `NovelAI Token` node

To get your token:
Open NovelAI → User Settings → Account → Get Persistent API Token.

⚠️ Requirements:
A valid NovelAI account is required. You must have either:
- an active subscription, or
- available Anlas balance

Designed for modular workflows, infinite generation setups, and both zero-cost workflows (no 💎 nodes) and advanced pipelines.

<img width="1458" height="1138" alt="novelai workflow" src="https://github.com/user-attachments/assets/a05cb6c6-9ace-4a84-9591-7bedd0e62709" />
