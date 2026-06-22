# Agentic AI & LLM Assisted Development Log (`docs/ax.md`)

This document details how our team leveraged advanced LLM coding assistants, generative engineering frameworks, and automated planning setups to rapidly build, debug, and optimize the AuraSync Edge Architecture for the Samsung EnnovateX AX Hackathon.

---

## 🤖 Agentic AI Workspace Configuration

During development, we established an iterative, agent-assisted feedback loop combining specialized software development LLMs with manual validation to navigate complex signal processing and hardware constraints. 

### 🔧 Workflows & Implementation Mechanics

1. **Architectural Ideation & Blueprinting**: We fed our initial raw hardware resource constraints ($<5\text{M}$ parameter limits, strict offline targets) through structural planning prompts. This helped us systematically eliminate heavy transformer models and isolate compliant open-weight acoustic candidates.
2. **PyTorch Quantization & Shape Debugging**: We utilized LLM assistants to generate clean tensor tracking shapes and troubleshoot compilation warnings when setting up the automated `torchaudio` dynamic resampling blocks.
3. **Multi-Threaded Path Engineering**: To bypass the limitations of the Python Global Interpreter Lock (GIL), coding assistants were used to generate asynchronous execution schemas, binding parallel audio buffers cleanly to separate Kaldi C++ worker paths.
4. **Model Context Protocol (MCP) Foundations**: We prototyped an experimental Model Context Protocol (`mcp_server.py`) inside the `assistant/backend/` architecture to manage semantic context sharing between local processes, utilizing structured context-handling blocks to isolate local telemetry data.

---

## 📈 Engineering Retrospective: What Worked and What Failed

### 🟢 What Worked (Successful Architectures)

* **The Move to ConvTasNet**: Originally, the pipeline was prototyped using deep Recurrent Neural Network (RNN) blocks. However, sequential recurrence drastically bottlenecked edge CPU memory allocation, causing real-time latency to spike to an unacceptable 12 seconds per stream. Using agentic code refactoring, we successfully pivoted to a fully parallelized 1D Convolutional Time-Domain Separation Matrix (ConvTasNet). This dropped active graph parameters down to 4.95M and achieved true real-time execution (`0.361 xRT`).
* **Deterministic Micro-Routing Over Semantic LLMs**: We tested a local 1-Billion parameter language model to parse text intent on-device. The inference pass added over 4 seconds of latency per command and frequently hallucinated device actions. Replacing this with a zero-parameter deterministic micro-routing dictionary structure (`Tier1IntentRouter`) eliminated latency and guaranteed zero-hallucination stability.
* **The 8kHz ↔ 16kHz Acoustic Bridge**: Running neural mask estimation directly on 16kHz audio required an unsustainable convolutional stride footprint. The LLM helped us design a decoupled logic: downsample to 8kHz for parallelized convolutional mask extraction, and then immediately upsample the resulting isolated channel back to 16kHz before hitting the Kaldi ASR engine. This preserved phoneme alignment while honoring hardware bounds.

### 🔴 What Failed (Dropped Approaches)

* **SuDoRM-RF Lite Matrix Integration**: We spent significant engineering time trying to implement an ultra-lightweight SuDoRM-RF acoustic model architecture. While it successfully reduced the parameter footprint, the lack of robust, pre-trained Indian English weight profiles caused massive mask bleeding, resulting in unreadable audio corruptions.
* **Dynamic PyTorch Level-1 Quantization**: Attempting to force 8-bit dynamic quantization on our convolutional separation matrix broke the backward compatibility of the internal tracking layers, leading to unstable tensor shapes during runtime inference. We abandoned this in favor of explicit graph optimizations and targeted channel downsampling via our Acoustic Bridge.
* **Untrained Generative Micro-Agents**: We originally designed a multi-agent orchestration setup (`supervisor.py`, `watchdog_critic.py`) using lightweight local LLMs to dynamically audit transcription quality. On edge CPU constraints, the context-handling memory overhead caused systemic buffer overflows in our stream-slicing ring buffers, proving that classic multi-agent text frameworks are too heavy for low-level edge gateway firmware.
