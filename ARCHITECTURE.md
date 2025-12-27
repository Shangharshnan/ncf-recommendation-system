# System Architecture

## High-Level Overview
The recommendation system is designed as a modular, production-ready pipeline that
supports real-time prediction, monitoring, and scalability.

## Architecture Diagram (Textual)

Client / User  
→ REST API (Flask)  
→ Data Preprocessing Layer (Pandas)  
→ Feature Engineering  
→ NCF Model (Inference)  
→ Ranking & Top-K Selection  
→ Response (JSON)

## Component Breakdown

### 1. Client Layer
- Sends user-item interaction data in JSON format
- Requests top-K recommendations

### 2. API Layer (Flask)
- Handles RESTful requests
- Validates input data
- Manages inference calls

### 3. Data Processing Layer
- Cleans raw data
- Handles missing values
- Encodes categorical features
- Applies real-time filtering

### 4. Model Layer (NCF)
- Learns non-linear user-item interactions
- Generates relevance scores
- Optimized for ranking tasks

### 5. Evaluation & Monitoring
- Tracks NDCG@K for recommendation quality
- Detects data drift using skewness and outlier analysis
- Flags retraining conditions

### 6. Storage Layer
- Stores interaction data in PostgreSQL / MySQL
- Maintains model artifacts using Pickle

### 7. Deployment Layer
- Dockerized application for environment consistency
- Scalable deployment ready

## Design Considerations
- Modular architecture for easy upgrades
- Separation of concerns between API, ML, and data layers
- Production-oriented monitoring and evaluation
