# Fine-Tuning NLLB for Low-Resource Odia-German Translation

This repository contains the complete source code, datasets, experimental results, and model artifacts for the thesis titled: **"Enhancing Contextual Understanding in Low-Resource Languages Using Multilingual Transformers"**

The research investigates and compares two primary methodologies for adapting a state-of-the-art multilingual model (`facebook/nllb-200-distilled-600M`) for a specialized, bidirectional Odia-German translation task in the journalistic domain: **Full Fine-Tuning (FFT)** and **Parameter-Efficient Fine-Tuning (PEFT) using LoRA**.

---

## Live Demonstrations

The final fine-tuned models and their interactive web applications are hosted on the Hugging Face Hub. Due to their large size, the model files are not stored in this GitHub repository.

* **Fully Fine-Tuned Model:**
    * **[Access the Model on Hugging Face Hub](https://huggingface.co/abhinandansamal/nllb-200-distilled-600M-finetuned-odia-german-bidirectional)**
    * **[Try the Live Web App](https://huggingface.co/spaces/abhinandansamal/full_fine_tuned_model_web_application)**

* **Adapter-Based (LoRA) Model:**
    * **[Access the Model on Hugging Face Hub](https://huggingface.co/abhinandansamal/nllb-200-distilled-600M-LoRA-finetuned-odia-german-bidirectional)**
    * **[Try the Live Web App](https://huggingface.co/spaces/abhinandansamal/Adapter_based_fine_tuned_model_web_application)**

---

## Dataset

The custom, human-validated parallel corpus created for this thesis is also available on the Hugging Face Hub:

* **[Access the Dataset on Hugging Face Hub](https://huggingface.co/datasets/abhinandansamal/bidirectional_odia_german_translation_parallel_corpus)**

---

## Repository Structure

This repository contains all the code and supporting files needed to reproduce the experiments and results.


      ├── 📂 data/
      │   ├── 📂 raw/
      │   │   ├── 📄 authentic_odia_corpus_v1.txt
      │   │   └── 📄 authentic_german_corpus_v1.txt
      │   └── 📂 transformed/
      │       ├── 📄 authentic_corpus,jsonl
      │       ├── 📄 authentic_corpus_final.jsonl
      │       ├── 📄 bidirectional_corpus.jsonl
      │       └── 📄 bidirectional_corpus_final.jsonl
      ├── 📂 images/
      │   └── 🖼️ (Plots, diagrams, and figures for the thesis)
      ├── 📂 notebooks/
      │   ├── 📜 odia_news_article_web_scraping.ipynb
      │   ├── 📜 check_num_of_lines.ipynb
      │   ├── 📜 final_data_instance_creation.ipynb
      │   ├── 📜 bidirectional_corpus_create.ipynb
      │   ├── 📜 bidirectional_full_fine_tuning_evaluation_NLLB.ipynb
      │   ├── 📜 bidirectional_LoRA_fine_tuning_evaluation_NLLB.ipynb
      │   ├── 📜 sampling_analysis_full_fine_tuning_LoRA_NLLB.ipynb
      │   ├── 📜 final_artifact_size.ipynb
      │   ├── 📜 model_deployment.ipynb
      │   ├── 📜 fully_fine_tuned_model_web_application.ipynb
      │   ├── 📜 Adapter_based_fine_tuned_model_web_application.ipynb
      │   ├── 📜 model_deployment.ipynb
      │   ├── 📜 nllb_200_distilled_600M.ipynb
      │   ├── 📜 data_upload_hf.ipynb
      │   ├── 📜 fully_fine_tuned_nllb_model_load_test.ipynb
      │   └── 📜 LoRA_fine_tuned_nllb_model_load_test.ipynb
      └── 📜 README.md


* **`data/`**: Contains the raw and transformed corpora.
* **`images/`**: Contains all the plots, figures, and diagrams generated for the thesis.
* **`notebooks/`**: Contains all the Jupyter/Colab notebooks in sequential order, from data collection to final analysis.

---

## Experimental Workflow

The research was conducted in a series of sequential steps, with each step documented in a corresponding Jupyter Notebook in the `/notebooks` directory.

1.  **Data Collection & Curation (`01` to `04`):** Raw articles were scraped from Odia newspapers, cleaned, validated, and transformed into a structured, bidirectional corpus.
2.  **Model Fine-Tuning (`05` & `06`):** Two separate experiments were conducted to fine-tune the base NLLB model: one using the traditional full fine-tuning approach and another using the parameter-efficient LoRA method.
3.  **Model Deployment (`07`):** The final model artifacts were uploaded to the Hugging Face Hub.
4.  **Analysis (`08` & `09`):** Final evaluations, efficiency calculations (artifact size), and advanced interpretability analyses (sampling) were performed.

---

## Key Results

This research conducted a comparative analysis of three models: the zero-shot Baseline, a Fully Fine-Tuned (FFT) model, and a parameter-efficient Adapter-Tuned (LoRA) model. The study found a clear and nuanced trade-off between the two fine-tuning methodologies, with the optimal strategy depending on the translation direction.

* For the `German → Odia` (Low-Resource Target) direction: The results revealed a fascinating trade-off. The Adapter-Tuned (LoRA) model was most effective at improving overall accuracy and fluency, achieving the highest BLEU score (26.33) and the best TER score (73.42). However, the Fully Fine-Tuned model proved superior in generating morphologically precise word-forms, attaining the highest chrF score (48.54). This suggests LoRA excels at learning sentence-level structure and vocabulary, while FFT is needed to master the most fine-grained grammatical details.

* For the `Odia → German` (High-Resource Target) direction: The Adapter-Tuned (LoRA) model was the unambiguous winner. It achieved the highest BLEU score (74.62) and the highest chrF score (82.33), significantly outperforming both the baseline and the fully fine-tuned model. Critically, it also matched the FFT model in producing the most fluent output, with both achieving an identical, best-in-class TER score of 39.39.

This research provides a valuable, data-driven case study for low-resource translation. It demonstrates that while the computationally expensive Full Fine-Tuning method can be superior for mastering the morphological complexity of a low-resource target language, the highly efficient LoRA methodology can deliver equivalent or even superior performance when translating into a high-resource language, making it a compelling and powerful adaptation strategy.

## Citation

If you use the code or datasets from this repository in your research, please cite the following thesis:

```bibtex
@mastersthesis{SamalThesis2025,
  author    = Abhinandan Samal,
  title     = Enhancing Contextual Understanding in Low-Resource Languages Using Multilingual Transformers,
  school    = IU International University of Applied Sciences,
  year      = 2025,
  url       = https://github.com/abhinandansamal/nllb_fine_tuned_odia_german_translator
}
