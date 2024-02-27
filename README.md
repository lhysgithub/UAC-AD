# UAC-AD
Source code for the paper "UAC-AD: Unsupervised Adversarial Contrastive Learning for Anomaly Detection on Multi-source Data"

### Environment
We support python3.x $\geq$ 3.7. The environment can be built by: ```$ pip install -r requirements.txt```

### Result records
The result records are in the `result21` directory.

### Reproducing UAD by running: 
`cd codes && python run.py`

### Experiment data types
Raw data for Dataset A: https://doi.org/10.5281/zenodo.7609780.
The metric types for Dataset A include CPU status, memory status, IO status, and network status.
The log type for Dataset A is Spark runtime logs.

Raw data for Dataset B: https://github.com/CloudWise-OpenSource/GAIA-DataSet/tree/main/MicroSS.
The Dataset B is mainly comes from a scenario in the business simulation system, MicroSS, owned by Cloudwise. It comes from a scenario of logging-in with QR Code.


The data type for Dataset C is restricted due to confidentiality requirements and is not disclosed at this time.

### Tree
```
.
├── README.md
├── codes
│   ├── common
│   │   ├── __init__.py
│   │   ├── data_loads.py
│   │   ├── data_processing.py
│   │   ├── data_processing_utils.py
│   │   ├── semantics.py
│   │   └── utils.py
│   ├── data_analysis.py
│   ├── gpu0.sh
│   ├── gpu1.sh
│   ├── models
│   │   ├── basev3.py
│   │   ├── fuse_v3.py
│   │   ├── kpi_model_v3.py
│   │   ├── log_model_v3.py
│   │   └── utils.py
│   └── run.py
├── data
│   └── chunk_10
│       ├── test.pkl
│       ├── train.pkl
│       ├── unlabel.pkl
│       └── unsupervised.pkl
├── requirements.txt
└── result21
    └── test.txt
```