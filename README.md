# Credit-Risk
A comprehensive Credit Risk solution: From mathematical modeling (PD, LGD, EAD, EL, PD, ) to containerized deployment simulating Cloud infrastructure (AWS S3) in an On-Premise environment.

![Status](https://img.shields.io/badge/Status-Deployed-success)
![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED)
![AWS](https://img.shields.io/badge/AWS-LocalStack_Simulation-orange)

## 📊 Dashboard Preview

<details>
<summary>Click to view screenshots</summary>

<p align="center">
  <img src="./assets/1.png" width="800"/>
  <img src="./assets/2.png" width="800"/>
  <img src="./assets/3.png" width="800"/>
  <img src="./assets/4.png" width="800"/>
  <img src="./assets/5.png" width="800"/>
  <img src="./assets/6.png" width="800"/>
</p>

</details>

## Overview

This project is a complete implementation of a **Credit Risk Decision Engine**, designed to simulate a real-world fintech environment.

While my previous projects focused on model training, here I wanted to tackle the **Engineering & MLOps** side of things. I moved beyond "notebook data science" to build a system where the model artifacts are decoupled from the application logic, simulating a Cloud Data Lake architecture on my own hardware.

The system calculates:
1.  **Probability of Default (PD)** using banking-standard methodologies.
2.  **Loss Given Default (LGD)** & **Exposure at Default (EAD)**.
3.  **Expected Loss (EL)** & **Expected Profit (EP)**.
4.  **Approval Decisions** based on dynamic profitability thresholds (Hurdle Rates), not just raw risk scores.

---

## 🏗️ Architecture & Infrastructure

The biggest challenge I set for myself was to simulate an AWS production environment without incurring cloud costs. To do this, I deployed the system on my Home Server using **Docker Compose** and **LocalStack**.

### The "Private Cloud" Setup
Instead of baking the `.pkl` model files inside the Docker image (which makes updates painful), I implemented an **Artifact Store pattern**:

1.  **S3 Emulation:** I run `LocalStack` to spin up a mock S3 bucket (`s3://credit-risk-models`).
2.  **Model Registry:** I use `awslocal` to sync my trained artifacts from the research environment to the bucket.
3.  **Inference Service:** When the FastAPI backend starts, it connects to this local S3 bucket to pull the latest model weights dynamically.

```mermaid
graph TD
    User((Credit Analyst)) -->|Browser| Frontend[Streamlit Frontend :8601]
    User -->|API Request| API[FastAPI Backend :8100]
    
    subgraph "Docker Network (Home Server)"
        Frontend -->|HTTP| API
        API -->|Inference| Pipeline[Credit Risk Pipeline]
        
        API -.->|Boto3| S3[LocalStack S3 Service :4566]
        Pipeline -.->|Load Artifacts| S3
        
        Bucket[(S3 Bucket:\ncredit-risk-models)] --- S3
    end
