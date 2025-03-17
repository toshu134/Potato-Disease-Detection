# Potato Disease Prediction API

This repository contains a FastAPI-based backend and a React-based frontend for predicting potato diseases using a trained deep learning model.
# dataset: https://www.kaggle.com/datasets/muhammadardiputra/potato-leaf-disease-dataset

## Overview
The Potato Disease Prediction API allows users to upload images of potato leaves and receive predictions about the type of disease affecting the leaves. The model can predict the following classes:
- Potato___Early_blight
- Potato___Late_blight
- Potato___healthy

This project is a potato leaf disease detection model. The current GitHub repo contains the FastAPI service code as well as the frontend website using ReactJS. The FastAPI server was uploaded on cloud through render cloud free tier hosting (https://render.com). The ReactJS app was hosted using Vercel free tier hosting (https://vercel.com/). 

## Features

- Image upload and disease prediction
- CORS enabled for frontend integration
- React-based frontend for user interaction

## Installation

### Backend

1. Clone the repository:
    ```sh
    git clone https://github.com/yourusername/potato-disease-prediction.git
    cd potato-disease-prediction
    ```

2. Create a virtual environment and activate it:
    ```sh
    python3 -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. Install the required packages:
    ```sh
    pip install -r requirements.txt
    ```

4. Ensure you have the trained model file (`new_model.keras`) in the project directory.

### Frontend

1. Navigate to the `frontend` directory:
    ```sh
    cd frontend
    ```

2. Install the required packages:
    ```sh
    npm install
    ```

## Usage

### Backend

1. Run the FastAPI server:
    ```sh
    uvicorn app:app --reload
    ```

2. The API will be accessible at `http://localhost:8000`.

### Frontend

1. Navigate to the `frontend` directory (if not already there):
    ```sh
    cd frontend
    ```

2. Start the React development server:
    ```sh
    npm start
    ```

3. The frontend will be accessible at `http://localhost:3000`.

## API Endpoints

### Root

- **GET /**

    Returns a welcome message.
    ```json
    {
        "test": "Welcome to the Potato Disease Prediction API"
    }
    ```

### Prediction

- **POST /prediction**

    Upload an image file for prediction.
    - **Request**:
        - `file`: Image file of the potato leaf.
    - **Response**:
        ```json
        {
            "class": "Potato___healthy",
            "confidence": 0.95
        }
        ```

## Frontend

The frontend is a React application that interacts with the backend API to upload images and display predictions. The frontend code is located in the `frontend` directory.
![image](https://github.com/user-attachments/assets/5c1eb0b2-ff50-49df-9945-7dc77e12f7dc)

## Project Structure

```
potato-disease-prediction/
│
├── API/
│   ├── app.py                # FastAPI application
│   ├── new_model.keras       # Trained model file
├── requirements.txt      # Python dependencies
├── frontend/             # React frontend
│   ├── public/
│   ├── src/
│   ├── node_modules
│   ├── package_lock.json
│   └── package.json
├── model_1.ipynb            # Jupyter notebooks
│    
└── README.md             # Project documentation
```

## Contributing

Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

