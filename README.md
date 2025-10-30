# STT (Speech-to-Text with Transformers)

This project implements a **Speech-to-Text (STT)** system built using **Transformers**.  
It provides modular components for audio preprocessing, model inference, and pipeline integration, making it easy to extend and adapt to different speech recognition tasks.

---

## 🚀 Features
- Speech-to-Text transcription using Transformer-based models.
- Modular architecture:
  - `components/` – Core building blocks
  - `models/` – Transformer-based speech recognition models
  - `pipeline/` – End-to-end pipeline for transcription
  - `utils/` – Helper functions
- Configurable settings in `configuration/`.
- Logging and exception handling built-in.

---

## 📂 Project Structure
STT/
│── cloud_storage/ # Cloud storage utilities
│── components/ # Core components of the STT system
│── configuration/ # Configuration files
│── constants/ # Constant values
│── entity/ # Data/entity definitions
│── exceptions/ # Custom exception handling
│── logger/ # Logging setup
│── models/ # Transformer models for STT
│── pipeline/ # Transcription pipeline
│── utils/ # Utility functions
│── templates/ # Templates (if needed for deployment/UI)
│── app.py # Main entry point
│── requirements.txt # Python dependencies
│── setup.py # Setup script

---

## ⚙️ Installation
Clone the repository and install dependencies:
```bash
git clone https://github.com/mohamedabbouda/STT.git
cd STT
pip install -r requirements.txt