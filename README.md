<div align="center">

# OmniForcing: Unleashing Real-time Joint Audio-Visual Generation

<p>
<a href="https://arxiv.org/abs/2603.11647"><img src="https://img.shields.io/badge/arXiv-2603.11647-b31b1b.svg" alt="arXiv"></a>
<a href="https://omniforcing.com"><img src="https://img.shields.io/badge/Project-Page-blue.svg" alt="Project Page"></a>
</p>

[Yaofeng Su]()\*¹ᐟ², [Yuming Li]()\*³, [Zeyue Xue]()¹ᐟ⁴, [Jie Huang]()¹, [Siming Fu]()¹, [Haoran Li]()¹, [Ying Li]()³, [Zezhong Qian]()³, [Haoyang Huang]()¹, [Nan Duan]()¹

¹JD Explore Academy &ensp; ²Fudan University &ensp; ³Peking University &ensp; ⁴The University of Hong Kong

\* Equal contribution

</div>

<div align="center">
<img src="static/images/teaser_2.png" width="100%">
</div>

**OmniForcing** is the first framework to distill an offline, bidirectional joint audio-visual diffusion model into a **real-time streaming autoregressive generator**. Built on top of LTX-2 (14B video + 5B audio), OmniForcing achieves **~25 FPS** streaming on a single GPU with a Time-To-First-Chunk of only **~0.7s** — a **~35× speedup** over the teacher — while maintaining visual and acoustic fidelity on par with the bidirectional teacher model.


## 📰 News

- **[2026/03]** Paper released on [arXiv 2603.11647](https://arxiv.org/abs/2603.11647). Project page is live at [https://omniforcing.com](https://omniforcing.com).
- **[Coming Soon]** Code and model weights will be open-sourced within two weeks. Stay tuned!


## 🔧 Method Overview

<div align="center">
<img src="static/images/method.png" width="100%">
</div>

OmniForcing employs a **three-stage distillation pipeline** to progressively transform the bidirectional teacher into a causal streaming engine:

- **Stage I — Bidirectional DMD:** Distribution Matching Distillation compresses the multi-step diffusion sampling into few-step denoising, while preserving the original global attention.

- **Stage II — Causal ODE Regression:** The model is equipped with our **Asymmetric Block-Causal Mask** and trained via ODE trajectory regression to adapt to causal attention. An **Audio Sink Token** mechanism with **Identity RoPE** is introduced to resolve the Softmax collapse and gradient explosion caused by extreme audio token sparsity.

- **Stage III — Joint Self-Forcing DMD:** The model autoregressively unrolls its own generations during training, enabling it to dynamically self-correct cumulative cross-modal errors from exposure bias.

At inference time, a **Modality-Independent Rolling KV-Cache** reduces per-step context complexity to O(L) and enables concurrent execution of the video and audio streams, achieving real-time synchronized generation.


## 📊 Results & Demos

### Main Results on JavisBench

| Model | Size | FVD ↓ | FAD ↓ | CLIP ↑ | AV-IB ↑ | DeSync ↓ | Runtime ↓ |
|:------|:----:|:-----:|:-----:|:------:|:-------:|:--------:|:---------:|
| MMAudio | 0.1B | – | 6.1 | – | 0.198 | 0.849 | 15s |
| JavisDiT++ | 2.1B | 141.5 | 5.5 | 0.316 | 0.198 | 0.832 | 10s |
| UniVerse-1 | 6.4B | 194.2 | 8.7 | 0.309 | 0.104 | 0.929 | 13s |
| LTX-2 (Teacher) | 19B | **125.4** | **4.6** | 0.318 | **0.318** | **0.384** | 197s |
| **OmniForcing (Ours)** | 19B | 137.2 | 5.7 | **0.322** | 0.269 | 0.392 | **5.7s** |

### Distillation Fidelity (VBench)

| Model | Aesthetic ↑ | Imaging ↑ | Motion Smooth. ↑ | Subject Consist. ↑ | TTFC ↓ | FPS ↑ |
|:------|:----------:|:---------:|:----------------:|:-----------------:|:------:|:-----:|
| LTX-2 (Teacher) | 0.569 | 0.574 | 0.993 | 0.945 | 197.0s | – |
| **OmniForcing** | **0.595** | **0.594** | **0.995** | **0.955** | **0.7s** | **25** |

### Demo Gallery

<table>
<tr>
<td align="center"><video src="static/videos/demo_ocean.mp4" width="280"></video><br><sub>Seaside — voice with bird calls</sub></td>
<td align="center"><video src="static/videos/demo_speech.mp4" width="280"></video><br><sub>Podium — sustained speech</sub></td>
<td align="center"><video src="static/videos/demo_sewing.mp4" width="280"></video><br><sub>Sewing — narration with machine sounds</sub></td>
</tr>
</table>

> 🔊 **Turn on sound!** Audio-visual synchronization is a key feature.
>
> For more demos and side-by-side comparisons with the LTX-2 teacher, visit our [Project Page](https://omniforcing.com).


## 🚀 Getting Started

> Coming Soon — code, model weights, and inference scripts will be released within two weeks.

<!--
### Installation
```bash
# TODO
```

### Inference
```bash
# TODO
```

### Training
```bash
# TODO
```
-->


## 📝 Citation

If you find OmniForcing useful in your research, please consider citing:

```bibtex
@article{su2026omniforcing,
  title   = {OmniForcing: Unleashing Real-time Joint Audio-Visual Generation},
  author  = {Su, Yaofeng and Li, Yuming and Xue, Zeyue and Huang, Jie and Fu, Siming
             and Li, Haoran and Li, Ying and Qian, Zezhong and Huang, Haoyang and Duan, Nan},
  journal = {arXiv preprint arXiv:2603.11647},
  year    = {2026}
}
```

## 🙏 Acknowledgements

OmniForcing builds upon several outstanding works. We thank the authors of [LTX-2](https://github.com/Lightricks/LTX-2), [Self-Forcing](https://self-forcing.github.io), [CausVid](https://github.com/tianweiy/CausVid), and [DMD](https://github.com/tianweiy/DMD2) for their pioneering contributions. The project page template is adapted from [Nerfies](https://nerfies.github.io/).
