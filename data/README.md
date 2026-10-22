# Data (not included)

The competition data is **not redistributed** in this repository. The FahMai RAG Challenge (Super AI Engineer Season 6, Level 1) was run on Kaggle by the organizers; get the files from the competition page or from the organizers, and read the competition rules before sharing anything derived from them.

What the notebook expects, under `MyDrive/RAG/data/` in Google Drive (or any folder you point the config cell at):

```
data/
├── questions.csv            100 rows: id, question, choice_1 ... choice_10
├── sample_submission.csv    id, answer (answer is an integer 1-10)
└── knowledge_base/
    ├── products/            110 Markdown files, one per product, named <CODE>_<brand>_<product>.md
    ├── policies/            5 files: cancellation, membership points, return, shipping, warranty
    └── store_info/          3 files: about the store, buying guide, general FAQ
```

The knowledge base is cut into 857 chunks and embedded with BGE-M3 (1024 dimensions). The notebook builds this index itself (a few minutes on a Colab GPU) and caches it in `MyDrive/RAG/cache_v3/`, so the first run is the slow one.

`results/submission_v5.csv` in this repository is the answer file that was submitted (question id and chosen option only).
