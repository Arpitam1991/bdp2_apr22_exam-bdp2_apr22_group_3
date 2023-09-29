import pymongo
import requests
import pandas as pd
import csv
import os
from dotenv import load_dotenv

def insert_json_data(data_list, collection_name, mongodb_uri, database_name):
    # Function to insert a list of JSON data into MongoDB collection
    client = pymongo.MongoClient(mongodb_uri)
    db = client[database_name]
    collection = db[collection_name]
    collection.insert_many(data_list)
    client.close()

# All urls and tokens
load_dotenv()

API_TOKEN = os.getenv("API_TOKEN")
mongodb_uri = os.getenv("MONGODB_URI")
database_name = os.getenv("DATABASE_NAME")
collection_name = os.getenv("COLLECTION_NAME")

response = requests.get("https://api.baserow.io/api/database/rows/table/199970/",
                        headers={"Authorization": key})

if response.status_code == 200:
    data = response.json()
    results = data['results']
    document_list = []


    for item in results:
        text = item.get('field_1368866', '')  # text
        label = item.get('field_1368867', '') # label

        if text and label:
            if label == 'undefined':
                label = '0'
            document_list.append({"text": text, "label": label})

    if document_list:
        csv_file_path = '../../api_data.csv'
        with open(csv_file_path, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = document_list[0].keys()
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            for document in document_list:
                writer.writerow(document)

        #Storing in mongodb
        insert_json_data(document_list, collection_name, mongodb_uri, database_name)
        print("Data stored in MongoDB.")
    else:
        print("List is empty. No data to insert.")

else:
    print(f'Error: {response.status_code}')


