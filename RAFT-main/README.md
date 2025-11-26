# RAFT — Customized Version for COMP 433 Project, mostly for the weather dataset

This repository contains a **clean, improved, and fully reproducible version** of the RAFT (Retrieval-Augmented Forecasting Transformer) model adapted for our COMP 433 project.  
It includes several **bug fixes**, **dataset handling improvements**, and **Colab-friendly modifications** so training is stable and easy.

---

###  Clean dataset pipeline
- Automatically detects and parses **date columns**
- Removes all **non-numeric** columns automatically  
- Ensures all data is **float32** (PyTorch requirement)
- Supports *any* numeric time-series CSV (Weather, Chess, etc.)
- Stable scaling + optional inverse-scaling

### Your CSV mist contain only numreic features
date,meantemp,humidity,wind_speed,meanpressure
2010-01-01,10.0,69,6.0,1015
2010-01-02,7.4,76,3.5,1017
...
