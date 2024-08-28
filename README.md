<div align="center">
<h3> Awesome AI (Machine Learning / Deep Learning) For DevSecOps</h3>
</div>

---

TODO...

<div align="center">
<img src="imgs/devsecops.png" alt="Paper Collection" width="600" style="vertical-align: middle;" />
</div>

---

## <img src="imgs/collection.png" alt="Paper Collection" width="50" style="vertical-align: middle;" /> Paper Collection

### <img src="imgs/plan.png" alt="Plan" width="30" style="vertical-align: middle;" /> Plan

- **Threat Modeling**
  - *No Relevant Publications Identified Using Our Defined Search Strategy*

- **Impact Analysis**
  - *No Relevant Publications Identified Using Our Defined Search Strategy*

### <img src="imgs/dev.png" alt="Dev" width="30" style="vertical-align: middle;" /> Development

- **Software Vulnerability Detection (SVD)**
  - **`Recurrent Neural Network (RNN)`**
    - Automatic feature learning for predicting vulnerable software components (TSE, 2018) [📝](https://ieeexplore.ieee.org/abstract/document/8540022)
    - Automated vulnerability detection in source code using deep representation learning (ICMLA, 2018) [📝](https://ieeexplore.ieee.org/abstract/document/8614145)
    - Vuldeepecker: A deep learning-based system for vulnerability detection (NDSS, 2018) [📝](https://arxiv.org/abs/1801.01681)
    - Vuldeelocator: a deep learning-based fine-grained vulnerability detector (TDSC, 2021) [📝](https://ieeexplore.ieee.org/abstract/document/9416836)
    - VUDENC: vulnerability detection with deep learning on a natural codebase for Python (IST, 2022) [📝](https://www.sciencedirect.com/science/article/pii/S0950584921002421)
  - **`Text Convolutional Neural Network (TextCNN)`**
    - A software vulnerability detection method based on deep learning with complex network analysis and subgraph partition (IST, 2023) [📝](https://www.sciencedirect.com/science/article/pii/S0950584923001830)
  - **`Graph Neural Network (GNN)`**
    - Devign: Effective vulnerability identification by learning comprehensive program semantics via graph neural networks (NeurIPS, 2019) [📝](https://proceedings.neurips.cc/paper_files/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html)
    - Bgnn4vd: Constructing bidirectional graph neural-network for vulnerability detection (IST, 2021) [📝](https://www.sciencedirect.com/science/article/pii/S0950584921000586)
    - Deep learning based vulnerability detection: Are we there yet (TSE, 2021) [📝](https://ieeexplore.ieee.org/abstract/document/9448435/)
    - Vulnerability detection with fine-grained interpretations (FSE, 2021) [📝](https://dl.acm.org/doi/abs/10.1145/3468264.3468597)
    - LineVD: Statement-level vulnerability detection using graph neural networks (MSR, 2022) [📝]()
    - mVulPreter: A Multi-Granularity Vulnerability Detection System With Interpretations (TDSC, 2022) [📝]()
    - VulChecker: Graph-based Vulnerability Localization in Source Code (USENIX, 2022) [📝]()
    - CPVD: Cross Project Vulnerability Detection Based On Graph Attention Network And Domain Adaptation (TSE, 2023) [📝]()
    - DeepVD: Toward Class-Separation Features for Neural Network Vulnerability Detection (ICSE, 2023) [📝]()
    - Learning Program Semantics for Vulnerability Detection via Vulnerability-Specific Inter-procedural Slicing (FSE, 2023) [📝]()
    - SedSVD: Statement-level software vulnerability detection based on Relational Graph Convolutional Network with subgraph embedding (IST, 2023) [📝]()
  - **`Node2Vec`**
    - Enhancing Deep Learning-based Vulnerability Detection by Building Behavior Graph Model (ICSE, 2023) [📝]()
  - **`Pre-trained Code Language Model (CLM) (Transformers)`**
    - Linevul: A transformer-based line-level vulnerability prediction (MSR, 2022) [📝](https://dl.acm.org/doi/abs/10.1145/3524842.3528452)
    - Vulnerability Detection by Learning from Syntax-Based Execution Paths of Code (TSE, 2023) [📝]()
  - **`LM + GNN`**
    - VELVET: a noVel Ensemble Learning approach to automatically locate VulnErable sTatements (SANER, 2022) [📝]()
    - Dataflow Analysis-Inspired Deep Learning for Efficient Vulnerability Detection (ICSE, 2023) [📝]()

<div align="center">
<b>Benchmarks used in evaluating AI-driven software vulnerability detection</b>
  
|                 Benchmark                | Year |  Granularity  | Programming Language | Real-World |  Synthesis |
|:----------------------------------------:|:----:|:-------------:|:--------------------:|:----------:|:----------:|
| [Firefox](https://link.springer.com/article/10.1007/s10664-011-9190-8)               | 2013 |      File     |        C, C++        | ✔ |            |
| [Android](https://ieeexplore.ieee.org/abstract/document/6860243/) | 2014 |      File     |         Java         | ✔ |            |
| [Draper](https://ieeexplore.ieee.org/abstract/document/8614145/)       | 2018 |    Function   |        C, C++        | ✔ | ✔ |
| [Vuldeepecker](https://arxiv.org/abs/1801.01681)   | 2018 |  Code Gadget  |        C, C++        | ✔ | ✔ |
| [Du et al.](https://ieeexplore.ieee.org/abstract/document/8812029/)                      | 2019 |    Function   |        C, C++        | ✔ |            |
| [Devign](https://proceedings.neurips.cc/paper_files/paper/2019/hash/49265d2447bc3bbfe9e76306ce40a31f-Abstract.html)             | 2019 |    Function   |        C, C++        | ✔ |            |
| [FUNDED](https://ieeexplore.ieee.org/abstract/document/9293321/)          | 2020 |    Function   |  C, Java, Swift, PHP | ✔ | ✔ |
| [Big-Vul](https://dl.acm.org/doi/abs/10.1145/3379597.3387501)                 | 2020 | Function/Line |        C, C++        | ✔ |            |
| [Reveal](https://ieeexplore.ieee.org/abstract/document/9448435/)        | 2021 |    Function   |        C, C++        | ✔ |            |
| [Cao et al.](https://www.sciencedirect.com/science/article/pii/S0950584921000586)                     | 2021 |    Function   |        C, C++        | ✔ |            |
| [D2A](https://ieeexplore.ieee.org/abstract/document/9402126/)                  | 2021 |    Function   |        C, C++        | ✔ |            |
| [Deepwukong](https://dl.acm.org/doi/abs/10.1145/3436877)    | 2021 |    Function   |        C, C++        | ✔ | ✔ |
| [Vuldeelocator](https://ieeexplore.ieee.org/abstract/document/9416836/) | 2021 |      Line     |        C, C++        | ✔ | ✔ |
| [VulCNN](https://dl.acm.org/doi/abs/10.1145/3510003.3510229)               | 2022 |    Function   |        C, C++        | ✔ | ✔ |
| [VUDENC](https://www.sciencedirect.com/science/article/pii/S0950584921002421)     | 2022 |     Token     |        Python        | ✔ |            |
| [DeepVD](https://ieeexplore.ieee.org/abstract/document/10172789/)             | 2023 |    Function   |        C, C++        | ✔ |            |
| [VulChecker](https://www.usenix.org/conference/usenixsecurity23/presentation/mirsky)   | 2023 |  Instruction  |        C, C++        | ✔ |            |
  
</div>

- **Software Vulnerability Classification (SVC)**
  - **`Machine Learning (ML)`**
    - Automation of vulnerability classification from its description using machine learning (ISCC, 2020) [📝]()
    - A machine learning approach to classify security patches into vulnerability types (CNS, 2020) [📝]()
  - **`RNN`**
    - Vuldeepecker: A deep learning-based system for vulnerability detection (NDSS, 2018) [📝](https://arxiv.org/abs/1801.01681)
    - μVulDeePecker: A Deep Learning-Based System for Multiclass Vulnerability Detection (TDSC, 2019) [📝]()
  - **`Text Recurrent Convolutional Neural Network (TextRCNN)`**
    - DeKeDVer: A deep learning-based multi-type software vulnerability classification framework using vulnerability description and source code (IST, 2023) [📝]()
  - **`Vanilla Transformer`**
    - Towards Vulnerability Types Classification Using Pure Self-Attention: A Common Weakness Enumeration Based Approach (CSE, 2021) [📝]()
  - **`Pre-trained Language Model (LM) (Transformers)`**
    - V2w-bert: A framework for effective hierarchical multiclass classification of software vulnerabilities (DSAA, 2021) [📝]()
    - Prediction of Vulnerability Characteristics Based on Vulnerability Description and Prompt Learning (SANER, 2023) [📝]()
  - **`CLM`**
    - VulExplainer: A Transformer-based Hierarchical Distillation for Explaining Vulnerability Types (TSE, 2023) [📝]()
    - AIBugHunter: A Practical tool for predicting, classifying and repairing software vulnerabilities (EMSE, 2023) [📝]()
  - **`CLM + RNN`**
    - Fine-grained commit-level vulnerability type prediction by CWE tree structure (ICSE, 2023) [📝]()

<div align="center">
<b>Benchmarks used in evaluating AI-driven software vulnerability classification</b>
  
|              Benchmark             | Year | Granularity |   Programming Language   | Real-World |  Synthesis |
|:----------------------------------:|:----:|:-----------:|:------------------------:|:----------:|:----------:|
| [μVulDeePecker](https://ieeexplore.ieee.org/abstract/document/8846081/) | 2019 | Code Gadget |          C, C++          | ✔ | ✔ |
| [TreeVul](https://ieeexplore.ieee.org/abstract/document/10172785/)         | 2023 |    Commit   | C, C++, Java, and Python | ✔ |            |
  
</div>

- **Automated Vulnerability Repair (AVR)**
  - **`ML`**
    - Sqlifix: Learning based approach to fix sql injection vulnerabilities in source code (SANER, 2021) [📝]()
  - **`CNN`**
    - Coconut: combining context-aware neural translation models using ensemble for program repair (ISSTA, 2020) [📝]()
  - **`RNN`**
    - Sequencer: Sequence-to-sequence learning for end-to-end program repair (TSE, 2019) [📝]()
    - A controlled experiment of different code representations for learning-based program repair (EMSE, 2022) [📝]()
  - **`Tree-based RNN`**
    - Dlfix: Context-based code transformation learning for automated program repair (ICSE, 2020) [📝]()
  - **`GNN`**
    - Hoppity: Learning graph transformations to detect and fix bugs in programs (ICLR, 2020) [📝]()
  - **`Vanilla Transformer`**
    - A syntax-guided edit decoder for neural program repair (FSE, 2021) [📝]()
    - Neural transfer learning for repairing security vulnerabilities in c code (TSE, 2022) [📝]()
    - Seqtrans: automatic vulnerability fix via sequence to sequence learning (TSE, 2022) [📝]()
    - Tare: Type-aware neural program repair (ICSE, 2023) [📝]()
  - **`CLM`**
    - Cure: Code-aware neural machine translation for automatic program repair (ICSE, 2021) [📝]()
    - Applying codebert for automated program repair of java simple bugs (MSR, 2021) [📝]()
    - Tfix: Learning to fix coding errors with a text-to-text transformer (PMLR, 2021) [📝]()
    - VulRepair: a T5-based automated software vulnerability repair (FSE, 2022) [📝]()
    - Improving automated program repair with domain adaptation (TOSEM, 2022) [📝]()
    - Vision Transformer-Inspired Automated Vulnerability Repair (TOSEM, 2023) [📝]()
    - Enhancing Code Language Models for Program Repair by Curricular Fine-tuning Framework (ICSME, 2023) [📝]()
    - Pre-trained model-based automated software vulnerability repair: How far are we? (TDSC, 2023) [📝]()
    - Examining zero-shot vulnerability repair with large language models (SP, 2023) [📝]()
    - Inferfix: End-to-end program repair with llms (FSE, 2023) [📝](https://dl.acm.org/doi/10.1145/3611643.3613892)
    - Unifying Defect Prediction, Categorization, and Repair by Multi-Task Deep Learning (ASE, 2023) [📝]()

<div align="center">
<b>Benchmarks used in evaluating AI-driven just-in-time (JIT) automated program/vulnerability repair</b>
  
|                 Benchmark                | Year | Programming Language | Real-World |  Synthesis |
|:----------------------------------------:|:----:|:--------------------:|:----------:|:----------:|
| [Defects4J](https://dl.acm.org/doi/abs/10.1145/2610384.2628055)       | 2014 |         Java         | ✔ |            |
| [ManyBugs](https://ieeexplore.ieee.org/abstract/document/7153570/)           | 2015 |           C          | ✔ |            |
| [BugAID](https://dl.acm.org/doi/abs/10.1145/2950290.2950308)       | 2016 |      JavaScript      | ✔ |            |
| [QuixBugs](https://dl.acm.org/doi/abs/10.1145/3135932.3135941)          | 2017 |     Java, Python     | ✔ |            |
| [CodeFlaws](https://ieeexplore.ieee.org/abstract/document/7965296/)        | 2017 |           C          | ✔ |            |
| [Bugs.jar](https://dl.acm.org/doi/abs/10.1145/3196398.3196473)             | 2018 |         Java         | ✔ |            |
| [SequenceR](https://ieeexplore.ieee.org/abstract/document/8827954/)       | 2019 |         Java         | ✔ |            |
| [Bugs2Fix](https://dl.acm.org/doi/abs/10.1145/3340544)      | 2019 |         Java         | ✔ |            |
| [ManySStuBs4J](https://dl.acm.org/doi/abs/10.1145/3379597.3387491) | 2020 |         Java         | ✔ |            |
| [Hoppity](https://par.nsf.gov/servlets/purl/10210341)        | 2020 |      JavaScript      | ✔ |            |
| [CodeXGLUE](https://arxiv.org/abs/2102.04664)         | 2021 |         Java         | ✔ | ✔ |
| [TFix](http://proceedings.mlr.press/v139/berabi21a.html)               | 2021 |      JavaScript      | ✔ |            |
| [VRepair](https://ieeexplore.ieee.org/abstract/document/9699412/)            | 2022 |        C, C++        | ✔ |            |
| [Namavar et al.](https://link.springer.com/article/10.1007/s10664-022-10223-5)              | 2022 |      JavaScript      | ✔ |            |
| [Pearce et al.](https://ieeexplore.ieee.org/abstract/document/10179324/)                | 2023 |        C, C++        | ✔ | ✔ |
| [Function-SStuBs4J](https://ieeexplore.ieee.org/abstract/document/10298436/)  | 2023 |         Java         | ✔ |            |
| [InferFix](https://dl.acm.org/doi/abs/10.1145/3611643.3613892)          | 2023 |       Java, C#      | ✔ |            |

</div>

- **Security Tools in IDEs**
  - **`LM-based Security Tool`**
    - AIBugHunter: A Practical tool for predicting, classifying and repairing software vulnerabilities (EMSE, 2023) [📝]()

### <img src="imgs/commit.png" alt="Commit" width="30" style="vertical-align: middle;" /> Code Commit

- **Dependency Management**
  - *No Relevant Publications Identified Using Our Defined Search Strategy*

- **CI/CD Secure Pipelines**
  - **`ML`**
    - Improving missing issue-commit link recovery using positive and unlabeled data (ASE, 2017) [📝]()
    - MULTI: Multi-objective effort-aware just-in-time software defect prediction (IST, 2018) [📝]()
    - Class imbalance evolution and verification latency in just-in-time software defect prediction (ICSE, 2019) [📝]()
    - Fine-grained just-in-time defect prediction (JSS, 2019) [📝]()
    - Effort-aware semi-supervised just-in-time defect prediction (IST, 2020) [📝]()
    - Just-in-time defect identification and localization: A two-phase framework (TSE, 2020) [📝]()
    - Adapting bug prediction models to predict reverted commits at Wayfair (FSE, 2020) [📝]()
    - JITLine: A simpler, better, faster, finer-grained just-in-time defect prediction (MSR, 2021) [📝]()
    - Enhancing just-in-time defect prediction using change request-based metrics (SANER, 2021) [📝]()
  - **`Explainable AI (XAI) For ML`**
    - Pyexplainer: Explaining the predictions of just-in-time defect models (ASE, 2021) [📝]()
  - **`RNN`**
    - DeepLink: Recovering issue-commit links based on deep learning (JSS, 2019) [📝]()
    - Deeplinedp: Towards a deep learning approach for line-level defect prediction (TSE, 2022) [📝]()
  - **`Tree-based RNN`**
    - Lessons learned from using a deep tree-based model for software defect prediction in practice (MSR, 2019) [📝]()
  - **`Vanilla Transformer`**
    - Deep just-in-time defect localization (TSE, 2021) [📝]()
  - **`LM`**
    - BTLink: automatic link recovery between issues and commits based on pre-trained BERT model (EMSE, 2023) [📝]()
  - **`CLM`**
    - EALink: An Efficient and Accurate Pre-trained Framework for Issue-Commit Link Recovery (ASE, 2023) [📝]() 
  - **`ML-based Just-In-Time (JIT) Software Defect Prediction (SDP) Tool`**
    - JITBot: an explainable just-in-time defect prediction bot (ASE, 2020) [📝]()
    - JITO: a tool for just-in-time defect identification and localization (FSE, 2020) [📝]()
  - **`ML-based Change Analysis Tool`**
    - Rex: Preventing bugs and misconfiguration in large services using correlated change analysis (USENIX, 2020) [📝]()

<div align="center">
<b>Benchmarks used in evaluating AI-driven just-in-time (JIT) software defect prediction</b>
  
|                Benchmark                | Year | Granularity |      Programming Language      | Real-World | Synthesis |
|:---------------------------------------:|:----:|:-----------:|:------------------------------:|:----------:|:---------:|
| [PROMISE](http://promise.site.uottawa.ca/SERepository/)    | 2007 |    Commit   |              Java              | ✔ |           |
| [Kamei et al.](https://ieeexplore.ieee.org/abstract/document/6341763/)                  | 2012 |    Commit   | C, C++, Java, JavaScript, Perl | ✔ |           |
| [Qt & OpenStack](https://dl.acm.org/doi/abs/10.1145/3180155.3182514)  | 2018 | Commit/Line |           C++, Python          | ✔ |           |
| [Cabral et al.](https://ieeexplore.ieee.org/abstract/document/8812072/)                 | 2019 | Commit/File |    Java, JavaScript, Python    | ✔ |           |
| [Yan et al.](https://ieeexplore.ieee.org/abstract/document/9026802/)                     | 2020 | Commit/File |              Java              | ✔ |           |
| [Wattanakriengkrai et al.](https://ieeexplore.ieee.org/abstract/document/9193975/) | 2020 |    Commit   |              Java              | ✔ |           |
| [Suh](https://dl.acm.org/doi/abs/10.1145/3368089.3417062)                 | 2020 | Commit/File |         JavaScript, PHP        | ✔ |           |
  
</div>

### <img src="imgs/test.png" alt="Test" width="30" style="vertical-align: middle;" /> Build, Test, and Deployment

- **Configuration Validation**
  - **`ML`**
    - Tuning configuration of apache spark on public clouds by combining multi-objective optimization and performance prediction model (JSS, 2021) [📝]()
    - KGSecConfig: A Knowledge Graph Based Approach for Secured Container Orchestrator Configuration (SANER, 2022) [📝]()
    - CoMSA: A Modeling-Driven Sampling Approach for Configuration Performance Testing (ASE, 2023) [📝]()
  - **`Feed-Forward Neural Network (FFNN)`**
    - DeepPerf: Performance prediction for configurable software with deep sparse neural network (ICSE, 2019) [📝]()
  - **`Generative Adversarial Network (GAN)`**
    - ACTGAN: automatic configuration tuning for software systems with generative adversarial networks (ASE, 2019) [📝]()
    - Perf-AL: Performance prediction for configurable software through adversarial learning (ESEM, 2020) [📝]()

- **Infrastructure Scanning**
  - **`ML`**
    - Characterizing defective configuration scripts used for continuous deployment (ICST, 2018) [📝]()
    - Source code properties of defective infrastructure as code scripts (IST, 2019) [📝]()
    - Within-project defect prediction of infrastructure-as-code using product and process metrics (TSE, 2021) [📝]()
  - **`Word2Vec-CBOW (Continuous Bag of Words)`**
    - FindICI: Using machine learning to detect linguistic inconsistencies between code and natural language descriptions in infrastructure-as-code (EMSE, 2022) [📝]()

<div align="center">
<b>Benchmarks used in evaluating AI-driven infrastructure as code</b>
  
|            Benchmark           | Year | Real-World |  Synthesis |
|:------------------------------:|:----:|:----------:|:----------:|
| [Rahman and Williams](https://ieeexplore.ieee.org/abstract/document/8367034/) | 2018 | ✔ |            |
| [Rahman and Williams](https://www.sciencedirect.com/science/article/pii/S0950584919300965)         | 2019 | ✔ |            |
| [Dalla et al.](https://ieeexplore.ieee.org/abstract/document/9321740/)          | 2021 | ✔ |            |
| [Borovits et al.](https://link.springer.com/article/10.1007/s10664-022-10215-5)      | 2022 |            | ✔ |
  
</div>

### <img src="imgs/monitor.png" alt="Monitor" width="30" style="vertical-align: middle;" /> Operation & Monitoring

- **Log Analysis & Anomaly Detection**
  - **`ML`**
    - An anomaly detection system based on variable N-gram features and one-class SVM (IST, 2017) [📝]()
    - Anomaly detection and diagnosis for cloud services: Practical experiments and lessons learned (JSS, 2018) [📝]()
    - Adaptive performance anomaly detection in distributed systems using online svms (TDSC, 2018) [📝]()
    - Log-based anomaly detection with robust feature extraction and online learning (TIFS, 2021) [📝]()
    - Try with Simpler--An Evaluation of Improved Principal Component Analysis in Log-based Anomaly Detection (TOSEM, 2023) [📝]()
    - On the effectiveness of log representation for log-based anomaly detection (EMSE, 2023) [📝]()
  - **`RNN`**
    - Deeplog: Anomaly detection and diagnosis from system logs through deep learning (CCS, 2017) [📝]()
    - Robust log-based anomaly detection on unstable log data (FSE, 2019) [📝]()
    - Loganomaly: Unsupervised detection of sequential and quantitative anomalies in unstructured logs (IJCAI, 2019) [📝]()
    - Anomaly detection in operating system logs with deep learning-based sentiment analysis (TDSC, 2020) [📝]()
    - SwissLog: Robust anomaly detection and localization for interleaved unstructured logs (TDSC, 2022) [📝]()
    - DeepSyslog: Deep Anomaly Detection on Syslog Using Sentence Embedding and Metadata (TIFS, 2022) [📝]()
    - LogOnline: A Semi-Supervised Log-Based Anomaly Detector Aided with Online Learning Mechanism (ASE, 2023) [📝]()
    - On the effectiveness of log representation for log-based anomaly detection (EMSE, 2023) [📝]()
  - **`RNN-based AutoEncoder (AE)`**
    - Lifelong anomaly detection through unlearning (CCS, 2019) [📝]()
    - Recompose event sequences vs. predict next events: A novel anomaly detection approach for discrete event logs (CCS, 2021) [📝]()
  - **`GNN`**
    - LogGraph: Log Event Graph Learning Aided Robust Fine-Grained Anomaly Diagnosis (TDSC, 2023) [📝]()
  - **`Vanilla Transformer`**
    - Log-based anomaly detection without log parsing (ASE, 2021) [📝]()
  - **`XAI For Deep Learning (DL)`**
    - Deepaid: Interpreting and improving deep learning-based anomaly detection in security applications (CCS, 2021) [📝]()
    - Towards an interpretable autoencoder: A decision-tree-based autoencoder and its application in anomaly detection (TDSC, 2022) [📝]()
  - **`Conditional Diffusion Model`**
    - Maat: Performance Metric Anomaly Anticipation for Cloud Services with Conditional Diffusion (ASE, 2023) [📝]()

<div align="center">
<b>Benchmarks used in evaluating AI-driven log analysis and anomaly detection</b>
  
|              Benchmark              | Year | Real-World |  Synthesis |
|:-----------------------------------:|:----:|:----------:|:----------:|
| [Yahoo! Webscope](https://webscope.sandbox.yahoo.com/)        | 2006 | ✔ | ✔ |
| [BGL](https://ieeexplore.ieee.org/abstract/document/4273008/) | 2007 | ✔ |            |
| [HDFS](https://dl.acm.org/doi/abs/10.1145/1629575.1629587)         | 2009 | ✔ |            |
| [ADFA-LD](https://ieeexplore.ieee.org/abstract/document/6555301/) | 2013 |            | ✔ |
| [SDS](https://ieeexplore.ieee.org/abstract/document/7371584/)            | 2015 |            | ✔ |
| [UNSW-NB15](https://ieeexplore.ieee.org/abstract/document/7348942/)   | 2015 |            | ✔ |
| [OpenStack](https://dl.acm.org/doi/abs/10.1145/3133956.3134015)      | 2017 | ✔ |            |
| [Microsoft](https://dl.acm.org/doi/abs/10.1145/3338906.3338931)    | 2019 | ✔ |            |
| [LogHub](https://github.com/logpai/loghub)          | 2020 | ✔ |            |
| [Studiawan et al.](https://ieeexplore.ieee.org/abstract/document/9259061/)          | 2020 | ✔ |            |
| [Yang et al.](https://dl.acm.org/doi/abs/10.1145/3644386)                   | 2023 | ✔ |            |
  
</div>

- **Cyber-Physical Systems**
  - **`ML`**
    - TABOR: A graphical model-based approach for anomaly detection in industrial control systems (CCS, 2018) [📝]()
    - Adaptive-Correlation-aware Unsupervised Deep Learning for Anomaly Detection in Cyber-physical Systems (TDSC, 2023) [📝]()
  - **`RNN + GNN`**
    - Digital Twin-based Anomaly Detection with Curriculum Learning in Cyber-physical Systems (TOSEM, 2023) [📝]()
  - **`GAN`**
    - Digital twin-based anomaly detection in cyber-physical systems (ICST, 2021) [📝]()
  - **`Variational AutoEncoder (VAE)`**
    - From Point-wise to Group-wise: A Fast and Accurate Microservice Trace Anomaly Detection Approach (FSE, 2023) [📝]()
  - **`Vanilla Transformer`**
    - Twin Graph-Based Anomaly Detection via Attentive Multi-Modal Learning for Microservice System (ASE, 2023) [📝]()
  - **`LM + RNN`**
    - KDDT: Knowledge Distillation-Empowered Digital Twin for Anomaly Detection (FSE, 2023) [📝]()

<div align="center">
<b>Benchmarks used in evaluating AI-driven log analysis and anomaly detection</b>
  
|                     Benchmark                    | Year | Real-World |  Synthesis |
|:------------------------------------------------:|:----:|:----------:|:----------:|
| [Gas Pipeline Dataset](https://sites.google.com/a/uah.edu/tommy-morris-uah/ics-data-sets) | 2015 | ✔ |            |
| [SWaT](https://ieeexplore.ieee.org/abstract/document/7469060/)                       | 2016 | ✔ |            |
| [WADI](https://dl.acm.org/doi/abs/10.1145/3055366.3055375)                        | 2017 | ✔ |            |
| [BATADAL](https://ascelibrary.org/doi/abs/10.1061/(ASCE)WR.1943-5452.0000969)                | 2018 |            | ✔ |
| [MSDS](https://ieeexplore.ieee.org/abstract/document/10025560/)                      | 2023 | ✔ |            |
  
</div>

---
