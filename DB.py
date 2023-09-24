import pymongo
import requests

def insert_json_data(data_list, collection_name, mongodb_uri, database_name):
    # Function to insert a list of JSON data into MongoDB collection
    client = pymongo.MongoClient(mongodb_uri)
    db = client[database_name]
    collection = db[collection_name]
    collection.insert_many(data_list)
    client.close()

# All urls and tokens
API_TOKEN = "PezlVLkQjEBjFM7OEBZnLjJRloYdJXKW"
key = 'Token' + " " + API_TOKEN

# DB
mongodb_uri = "mongodb+srv://SubhLodh:subhlodh@cluster0.yc9it6o.mongodb.net/"
database_name = "tweets"
collection_name = "tweet_data"

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
        insert_json_data(document_list, collection_name, mongodb_uri, database_name)
        print("Data stored in MongoDB.")
    else:
        print("List is empty. No data to insert.")

else:
    print(f'Error: {response.status_code}')
