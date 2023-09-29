# Project Name
Twitter Sentiment Analysis



## Data Collection

We obtained the dataset from Kaggle, a prominent platform for data sharing and analysis. 

## Data Storage

To efficiently manage and store our dataset, we leveraged MongoDB. The data storage process involved the use of a custom Python script (`db.py`) for data ingestion and organization. 

## Sentiment Analysis

The heart of our project revolves around sentiment analysis. We extracted data from MongoDB and performed extensive preprocessing to prepare the data for sentiment analysis. Our sentiment analysis model,main.py ,  and then saved as `model.h5`.

## User Interface

We developed a user-friendly interface using DashApplication, a Python web framework. Users can interact with the interface to perform sentiment analysis on their own text inputs.

## Docker Image

We containerized our project using Docker for seamless deployment and scaling.

## Deployment

For production deployment, we used AWS, we used aws CLI services to deploy docker image. we used AWS ECr to push the image onto our Repository.

## Data Version Control (DVC)
To maintain data integrity and version control, we integrated Data Version Control (DVC) into our project workflow. DVC allows us to track changes and collaborate effectively on data-related tasks.

## References
For more information on the tools and technologies used in this project, you can refer to the following resources:

Kaggle: Kaggle, a valuable source for datasets and data science competitions.
MongoDB: MongoDB, a NoSQL database used for efficient data storage and retrieval.
Dash by Plotly: Dash, a Python web framework for building interactive web applications.
Docker: Docker, a platform for developing, shipping, and running applications in containers.
Kubernetes: Kubernetes, an open-source container orchestration platform.



## Requirements

To set up and run the project locally, ensure you have the necessary dependencies installed. You can install them using the following command:

pip install -r requirements.txt
