# Products Chatbot - FastAPI Backend

A modern, fast, and scalable FastAPI backend for the Products Chatbot project.

## Setup & Installation

Follow these steps to set up and run the project locally.

### Prerequisites

Ensure you have [Conda](https://docs.conda.io/en/latest/) installed on your system.

### 1. Create Conda Environment
Create a new conda environment named `products-chatbot` with Python 3.12:
```bash
conda create -n products-chatbot python=3.12 -y
```

### 2. Activate the Environment
To activate the created environment, run:
```bash
conda activate products-chatbot
```

### 3. Install Dependencies
Once the environment is active, install the project dependencies:
```bash
pip install -r requirements.txt
```
*(Note: Create a `requirements.txt` file if it doesn't exist yet, or install packages like `fastapi` and `uvicorn` manually).*

### 4. Running the Application
To start the FastAPI development server:
```bash
uvicorn src.main:app --reload
```
