import pymongo

host_string = "mongodb://localhost"

# this is the default port for mongodb
port = 27017

mongo_client = pymongo.MongoClient(host_string, port)

# get a reference to the mongodb database 'test'
mongo_db = mongo_client['test']

# get a reference to the 'user profiles' collection in the 'test' database
user_profiles_collection = mongo_db['user_profiles']

user_profiles_collection.insert(friends_profiles)
user_profiles_collection.insert(followers_profiles)

# putting the above lines into a function
def save_json_data_to_mongo(data, mongo_db,
                            mongo_db_collection,
                            host_string = "localhost",
                            port = 27017):
    mongo_client = pymongo.MongoClient(host_string, port)
    mongo_db = mongo_client[mongo_db]
    collection = mongo_db[mongo_db_collection]
    inserted_object_ids = collection.insert(data)
    return(inserted_object_ids)