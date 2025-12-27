# Recommendation System Architecture – NCF

## Overview
This project presents the design and planned implementation of an end-to-end recommendation system using
Neural Collaborative Filtering (NCF) to model non-linear user–item interactions. The system is designed
to serve real-time recommendations through a RESTful API and support scalable deployment.

## Problem Statement
Traditional collaborative filtering methods struggle to capture complex, non-linear user-item
interaction patterns. This project aims to design a deep learning–based recommendation system
that improves ranking relevance while remaining production-ready.

## Proposed Solution
The system leverages Deep Neural Collaborative Filtering (NCF) to learn latent representations of
users and items. A Flask-based API receives raw interaction data in JSON format, performs real-time
data preprocessing, and returns ranked recommendations.

Model quality is monitored using drift analysis techniques to ensure prediction reliability
over time.

## System Architecture (High Level)

User / Client  
→ REST API (Flask)  
→ Data Processing (Pandas, Filtering, Feature Engineering)  
→ NCF Model (Inference)  
→ Recommendation Output (Top-K Items)

## Model Evaluation
- Metric: **NDCG@K**
- Focus on ranking relevance and recommendation quality

## Drift & Quality Monitoring
- Skewness-based feature drift detection
- Outlier detection for incoming data
- Automated retraining triggers (planned)

## Tech Stack
- Language: Python
- Deep Learning Framework: PyTorch
- Model: Deep Neural Collaborative Filtering (NCF)
- Backend: Flask (REST API)
- Database: PostgreSQL / MySQL
- Data Processing: Pandas
- Deployment: Docker
- Model Serialization: PyTorch (`.pt`) / Pickle

## Implementation Plan
- Phase 1: Dataset preparation and baseline NCF model
- Phase 2: REST API for prediction service
- Phase 3: Drift detection and monitoring
- Phase 4: Dockerization and deployment

## Status
Design and architecture completed. Implementation planned and in progress.
