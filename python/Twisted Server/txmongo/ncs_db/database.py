import _local_path
import sys
import time

from twisted.internet import defer, reactor
from twisted.python import log
from bson import ObjectId

import txmongo

DEBUG = True

''' This class is the object through which all database interaction takes place. MongoDB is the 
    underlying database and PyMongo cannot be used with Twisted because the function calls are 
    blocking. An alternate interface called TxMongo that provides asynchronous functions is used.'''
class TxMongoDatabase():

    def __init__(self, db_ref):

        try:
            self.connection = txmongo.MongoConnection()
            self.db_ref = self.connection.db_ref
            self.query_results = None
        except Exception, e:
            log.msg("Database connection error: " + str(e))
            if DEBUG:
                print "Database connection error: " + str(e)


    ''' This decorator is used with generator functions as an alternative method of using the 
        Deferred object with nested callbacks. The generator yield functions allow for creating
        a series of asynchronous callbacks (the function will not restart after the yield until the
        deferred has finished execution) '''

    # The inline callbacks still run one at a time and they are still invoked by the reactor.
    # To see the order in which they are called, put print statements in each of the callbacks.
    # TO RETURN SOMETHING MEANINGFUL: defer.returnValue(poem) RETURN RESULT?
    # ADD AUTHENTICATION TO ENTIRE DB SO DAEMON IS THE ONLY ONE THAT CAN CONNECT TO IT (SEE DATABASE.PY IN TXMONGO)

    # SEE COLLECTION.PY FOR IMPLEMENTED WRAPPER FUNCTIONS
    @defer.inlineCallbacks
    def insert(self, collection, data):
        if DEBUG:
            print 'Inserting document into collection ' + collection

        try:
            result = yield self.db_ref[collection].insert(data, safe=True)
        except Exception, e:
            log.msg("Insert error: " + str(e))
            if DEBUG:
                print 'Insert error: ' + str(e)

        if DEBUG:
            print '\n listing collections:'
            colls = yield self.db_ref.collection_names()
            for coll in colls:
                print coll

    # returns all documents in a collection
    @defer.inlineCallbacks
    def query(self, collection):
        if DEBUG:
            print 'Querying collection ' + collection        

        try:
            #f = txmongo.filter.sort(txmongo.filter.DESCENDING("test"))
            self.query_results = yield self.db_ref[collection].find()
            if DEBUG:
                print 'Documents found: '
                for doc in self.query_results:
                    print doc
            defer.returnValue(self.query_results)
        except Exception, e:
            log.msg("Query error: " + str(e))
            if DEBUG:
                print 'Query error: ' + str(e)

    # returns the documents with the specified field equaling some value
    @defer.inlineCallbacks
    def query_by_field(self, collection, field):
        if DEBUG:
            print 'Querying by collection ' + collection + ' by ' + str(field)

        try:
            self.query_results = yield self.db_ref[collection].find(field)
            if DEBUG:
                print 'Documents found: '
                for doc in self.query_results:
                    print doc
            defer.returnValue(self.query_results)
        except Exception, e:
            log.msg("Query by field error: " + str(e))
            if DEBUG:
                print 'Query by field error: ' + str(e)

    # replaces a document but does NOT insert the document if it cannot be found
    @defer.inlineCallbacks
    def update_doc(self, collection, doc, new_doc):
        if DEBUG:
            print 'Updating collection'       

        try:
            result = yield self.db_ref[collection].update(doc, new_doc, safe=True)
            if DEBUG:
                print 'Update result: ' + str(result)
        except Exception, e:
            log.msg("Update error: " + str(e))
            if DEBUG:
                print 'Update error: ' + str(e)

    @defer.inlineCallbacks
    def remove_coll(self, collection):
        if DEBUG:
            print 'Removing collection ' + collection

        try:
            result = yield self.db_ref[collection].drop(safe=True)
        except Exception, e:
            log.msg("Remove error: " + str(e))
            if DEBUG:
                print 'Remove error: ' + str(e)

    # drop all collections and close connection
    @defer.inlineCallbacks
    def drop(self):
        if DEBUG:
            print 'Dropping database'
        try:
            colls = yield self.db_ref.collection_names()
            for coll in colls:
                self.remove_coll(coll)
        except Exception, e:
            log.msg("Drop error: " + str(e))
            if DEBUG:
                print 'Drop error: ' + str(e)        
        try:
            yield self.connection.disconnect()
        except Exception, e:
            log.msg("Disconnect error: " + str(e))
            if DEBUG:
                print 'Disconnect error: ' + str(e)

# FIX DROP DATABASE FUNCTION, SO LIKE, IT ACTUALLY WORKS

# this is only used for testing
if __name__ == '__main__':

    '''db = TxMongoDatabase("foo")

    # inline callbacks return a deferred type
    d = db.insert("test", {"something": 10, "somethingelse": "nothing"})
    d.addCallback(lambda ign: db.insert("users", {"blah": "testing"}))
    d.addCallback(lambda ign: db.query("test"))
    d.addCallback(lambda ign: db.query_by_field("test", {"something": 10}))
    d.addCallback(lambda ign: db.update_doc("test", {"something": 10}, {"something": 12}))
    d.addCallback(lambda ign: db.query_by_field("test", {"something": 10}))
    d.addCallback(lambda ign: db.query_by_field("test", {"something": 12}))
    d.addCallback(lambda ign: db.remove_coll("test"))
    d.addCallback(lambda ign: db.query("test"))
    d.addCallback(lambda ign: db.drop())'''

    db = TxMongoDatabase("ncb")

    #d = db.db_ref["testuser@gmail.com_models"].drop(safe=True)

    '''d = db.insert("users", 
        {
            "username": "testuser@gmail.com",
            "password": "supersecretpassword",
            "first_name": "Hersheys",
            "last_name": "Bar",
            "institution": "UNR",
            "salt": None
        })'''

    d = db.query("users")

    '''d.addCallback(lambda ign: db.insert("users", 
        {
            "username": "testuser@gmail.com",
            "password": "supersecretpassword",
            "first_name": "Hersheys",
            "last_name": "Bar",
            "institution": "UNR",
            "lab_id": 8675309,
            "salt": None,
            "models": None
        }))'''

    '''d.addCallback(lambda ign: db.insert("global_models", 
    {
        "name": "test model",
        "author": "Wesley Snipes",
        "date": "05-12-2015"
    }))'''


    rev_0 = {
        "date_created": time.strftime("%m/%d/%Y"),
        "last_modified": time.strftime("%m/%d/%Y"),
        "model": 
            {
            "author": "Hersheys Bar", 
            "cellAliases": [], 
            "cellGroups": {
              "cellGroups": [
                {
                  "hashKey": "09B", 
                  "classification": "cells", 
                  "description": "Description", 
                  "geometry": "Sphere", 
                  "name": "Cell 3", 
                  "num": 150, 
                  "parameters": {
                    "a": {
                      "maxValue": 0, 
                      "mean": 0, 
                      "minValue": 0, 
                      "stddev": 0, 
                      "type": "exact", 
                      "value": 0.2
                    }, 
                    "b": {
                      "maxValue": 0, 
                      "mean": 0, 
                      "minValue": 0, 
                      "stddev": 0, 
                      "type": "exact", 
                      "value": 0.2
                    }, 
                    "c": {
                      "maxValue": 0, 
                      "mean": 0, 
                      "minValue": 0, 
                      "stddev": 0, 
                      "type": "exact", 
                      "value": -65
                    }, 
                    "d": {
                      "maxValue": 0, 
                      "mean": 0, 
                      "minValue": 0, 
                      "stddev": 0, 
                      "type": "exact", 
                      "value": 8
                    }, 
                    "threshold": {
                      "maxValue": 0, 
                      "mean": 0, 
                      "minValue": 0, 
                      "stddev": 0, 
                      "type": "exact", 
                      "value": 30
                    }, 
                    "type": "Izhikevich", 
                    "u": {
                      "maxValue": -11, 
                      "mean": 0, 
                      "minValue": -15, 
                      "stddev": 0, 
                      "type": "uniform", 
                      "value": 0
                    }, 
                    "v": {
                      "maxValue": -55, 
                      "mean": 0, 
                      "minValue": -75, 
                      "stddev": 0, 
                      "type": "uniform", 
                      "value": 0
                    }
                  }
                }
              ], 
              "classification": "cellGroup", 
              "description": "Description", 
              "name": "Home"
            }, 
            "classification": "model", 
            "description": "Description", 
            "name": "Test Model", 
            "synapses": []
          }
    }

    new_model = {
        "model_rev_0": rev_0,
        "model_rev_1": rev_0
    }


    #d.addCallback(lambda ign: db.insert("testuser@gmail.com_models", new_model))

    #d.addCallback(lambda ign: db.query("testuser@gmail.com_models"))

    changed_model = new_model.copy()
    print 'testing:'
    print changed_model['model_rev_1']['model']['description'] + '\n\n\n'
    changed_model['model_rev_1']['model']['description'] = 'a fine model indeed'
    #del changed_model['model_rev_0']

    d.addCallback(lambda ign: db.query("testuser@gmail.com_models"))
    #d.addCallback(lambda ign: db.update_doc("testuser@gmail.com_models", new_model, changed_model))
    #d.addCallback(lambda ign: db.update_doc("testuser@gmail.com_models", {'_id': ObjectId('557710a81d41c8799ca8cb3a')}, changed_model))
    #d.addCallback(lambda ign: db.query("testuser@gmail.com_models"))
    #d.addCallback(lambda ign: db.query("global_models"))
    #d = db.query("users")

    field1 = {'model_rev_1.model.author': 'Hersheys Bar'}
    field2 = {'model_rev_1.model.name': 'Test Model'}
    #d.addCallback(lambda ign: db.query_by_field('testuser@gmail.com_models', {'$and': [field1, field2]}))

    # kill reactor loop after this callback series
    d.addCallback(lambda ign: reactor.stop())

    reactor.run()