---
language:
- en
multilinguality:
- monolingual
size_categories:
- 100K<n<1M
task_categories:
- feature-extraction
- sentence-similarity
pretty_name: Natural Questions
tags:
- sentence-transformers
dataset_info:
  config_name: pair
  features:
  - name: query
    dtype: string
  - name: answer
    dtype: string
  splits:
  - name: train
    num_bytes: 67154228
    num_examples: 100231
  download_size: 43995757
  dataset_size: 67154228
configs:
- config_name: pair
  data_files:
  - split: train
    path: pair/train-*
---

# Dataset Card for Natural Questions

This dataset is a collection of question-answer pairs from the Natural Questions dataset. See [Natural Questions](https://ai.google.com/research/NaturalQuestions) for additional information.
This dataset can be used directly with Sentence Transformers to train embedding models.

## Dataset Subsets

### `pair` subset

* Columns: "question", "answer"
* Column types: `str`, `str`
* Examples:
    ```python
    {
      'query': 'the si unit of the electric field is',
      'answer': 'Electric field An electric field is a field that surrounds electric charges. It represents charges attracting or repelling other electric charges by exerting force.[1] [2] Mathematically the electric field is a vector field that associates to each point in space the force, called the Coulomb force, that would be experienced per unit of charge, by an infinitesimal test charge at that point.[3] The units of the electric field in the SI system are newtons per coulomb (N/C), or volts per meter (V/m). Electric fields are created by electric charges, and by time-varying magnetic fields. Electric fields are important in many areas of physics, and are exploited practically in electrical technology. On an atomic scale, the electric field is responsible for the attractive force between the atomic nucleus and electrons that holds atoms together, and the forces between atoms that cause chemical bonding. The electric field and the magnetic field together form the electromagnetic force, one of the four fundamental forces of nature.',
    }
    ```
* Collection strategy: Reading the NQ train dataset from [embedding-training-data](https://huggingface.co/datasets/sentence-transformers/embedding-training-data).
* Deduplified: No