# -*- coding: utf-8 -*-

import optparse, os
from twisted.internet import defer
from twisted.internet.defer import Deferred, succeed
from twisted.internet.protocol import ServerFactory, Protocol
from twisted.application import internet, service
from twisted.python import log
import json
import bcrypt

import sys
sys.path.append('txmongo/ncs_db')
from database import TxMongoDatabase

DEBUG = True

#class userAlreadyExists(Exception): pass

class AddUserProtocol(Protocol):

    service = None
    deferred = None

    # called when a message is received from the client
    def dataReceived(self, data):

        # deserialize the data
        message = json.loads(data)

        # TODO: check schema

        if "request" in message:
            if message.get("request") == "addUser":

                # check if user already exists
                query = {}
                query['username'] = message['user'].get('username')
                self.deferred = self.service.db.query_by_field("users", query)

                # add user if it does not exist
                user_params = message['user']
                deferred = Deferred()
                self.deferred.addCallbacks(self.service.add_user, self.service.add_user_error, callbackArgs=(user_params, deferred), errbackArgs=(user_params, deferred))

                # send success or error message and close the connection
                self.deferred.addBoth(self.transport.write)
                self.deferred.addBoth(lambda ign: self.transport.loseConnection())

            else:
                self.transport.write('Invalid request. Goodbye.')
                self.transport.loseConnection()
        else:
            self.transport.write('Invalid request format. Goodbye.')
            self.transport.loseConnection()

class AddUserProtocolFactory(ServerFactory):

    protocol = AddUserProtocol

    def __init__(self, service):
        self.protocol.service = service

class AddUserService(service.Service):

    def __init__(self):
        self.db = TxMongoDatabase("ncb")

    def startService(self):
        service.Service.startService(self)
        log.msg('Service for adding new users has been started')

    def add_user(self, query_results, user_params, deferred):

        if DEBUG:
            print 'In add user callback'

        if query_results:

            log.msg('Add new user failed. Username already taken.')

            if DEBUG:
                print 'Query results:'
                for doc in query_results:
                    print doc

            # notify client of error
            return 'Add new user failed.'

        else:

            # encrypt password
            print user_params['password']
            salt = bcrypt.gensalt()
            user_params['password'] = bcrypt.hashpw(str(user_params['password']), salt)
            user_params['salt'] = salt

            if DEBUG:
                print 'Valid username. Inserting new user.'
                print 'Encrypted password: ' + str(user_params['password'])
                print 'SALT: ' + str(user_params['salt'])

            # insert user into database
            self.db.insert("users", user_params)

            # notify client of successful new user
            return 'Add new user successful.'

    def add_user_error(self, query_results, user_params, deferred):
        if DEBUG:
            print 'In add user errorback'
        return 'Add new user failed.'