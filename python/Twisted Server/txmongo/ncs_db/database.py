import _local_path
import sys
import time

from twisted.internet import defer, reactor
from twisted.python import log

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
            print 'Inserting collection'

        try:
            result = yield self.db_ref.collection.insert(data, safe=True)
        except Exception, e:
            log.msg("Insert error: " + str(e))
            if DEBUG:
                print 'Insert error: ' + str(e)

    # this function is mostly for testing
    # returns all the documents in the database (collection ignored?)
    # HOW ON EARTH DO YOU QUERY BY COLLECTION?
    @defer.inlineCallbacks
    def query(self, collection):
        if DEBUG:
            print 'Querying collection ' + collection        

        # fetch some documents (but not too many)
        try:
            #f = txmongo.filter.sort(txmongo.filter.DESCENDING("test"))
            docs = yield self.db_ref.collection.find(limit=100)
            if DEBUG:
                print 'Documents found: '
                for doc in docs:
                    print doc
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
            docs = yield self.db_ref.collection.find(field)
            if DEBUG:
                print 'Documents found: '
                for doc in docs:
                    print doc
            defer.returnValue(docs)
        except Exception, e:
            log.msg("Query by field error: " + str(e))
            if DEBUG:
                print 'Query by field error: ' + str(e)

    # this does NOT insert the document if it cannot be found
    @defer.inlineCallbacks
    def update_doc(self, collection, field, new_field):
        if DEBUG:
            print 'Updating collection'       

        try:
            result = yield self.db_ref.collection.update(field, {"$set": new_field}, safe=True)
        except Exception, e:
            log.msg("Update error: " + str(e))
            if DEBUG:
                print 'Update error: ' + str(e)

    @defer.inlineCallbacks
    def remove_coll(self, collection):
        if DEBUG:
            print 'Removing collection'

        try:
            result = yield self.db_ref.collection.drop(safe=True)
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
                yield self.db_ref.coll.drop()
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

# ADD A FUNCTION FOR STORING NEW USERS (IN THIS FILE?)

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
    d = db.insert("users", 
        {
            "username": "testuser@gmail.com",
            "password": "supersecretpassword",
            "first_name": "Hersheys",
            "last_name": "Bar",
            "institution": "UNR",
            "salt": None
        })
    d.addCallback(lambda ign: db.query("users"))

    #d = db.query("users")

    # kill reactor loop after this callback series
    d.addCallback(lambda ign: reactor.stop())

    reactor.run()