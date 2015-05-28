# -*- coding: utf-8 -*-

import optparse, os
from twisted.internet.defer import Deferred, succeed
from twisted.internet.protocol import ClientFactory, ServerFactory, Protocol
from twisted.application import internet, service
from twisted.python import log
import json
import sys, ncs

# static port for receiving streamed data from NCS
simOutputRecvPort = 8005

class SendSimDataToClientProtocol(Protocol):
    '''This is the class that receives the sim data from the simulator.
    It concats the data until the simulator drops the connection (because its done)'''

    totalData = ''

    # This is where the data is streaming in (can write to file or out here?)
    def dataReceived(self, data):
        self.totalData += data    #THIS SHOULD BE WRITTEN OUT OR IT MIGHT OVERFLOW

    # this depends on the fact that the sim closes the connection when it is done sending
    def connectionLost(self, reason):
        self.fullStreamReceived(self.totalData)

    def fullStreamReceived(self, totalData):
        self.factory.stream_finished(totalData)

# this factory just stores data each protocol instance sends to a client
class SendSimDataToClientProtocolFactory(ClientFactory):

    protocol = SendSimDataToClientProtocol

    def __init__(self):
        self.deferred = Deferred()

    def stream_finished(self, data):
        if self.deferred is not None:
            d, self.deferred = self.deferred, None
            d.callback(data)

    # this isn't called now, but it could be called from
    # the dataReceived function above instead of queuing
    # up the data received before sending it
    def data_received(self, data):
        if self.deferred is not None:
            d, self.deferred = self.deferred, None
            d.callback(data)

    def clientConnectionFailed(self, connector, reason):
        if self.deferred is not None:
            d, self.deferred = self.deferred, None
            d.errback(reason)

class RecvSimDataProxyProtocol(Protocol):

    # a callback that is invoked after a client opens a connection
    def connectionMade(self):

        deferred = self.factory.service.recv_data()

        # send the client the data when we receive it
        deferred.addCallback(self.transport.write)

        # this closes the connection (once the callbacks are finished)
        # we'll leave this here until we change it so the proxy sends data
        # as it gets it (so it shouldn't close the connection)
        deferred.addBoth(lambda r: self.transport.loseConnection())

# this is like the ProxyServiceFactory
class RecvSimDataProxyProtocolFactory(ServerFactory):

    protocol = RecvSimDataProxyProtocol

    def __init__(self, service):
        self.service = service

class ProxyService(service.Service):

    data = None

    def __init__(self):
        # set recv factory instance?
        pass

    def startService(self):
        service.Service.startService(self)
        log.msg('Service for receiving sim data is running')

    def recv_data(self):
        clientFactory = SendSimDataToClientProtocolFactory()

        # this adds the callback of getting the data to the client's deferred
        clientFactory.deferred.addCallback(self.set_data)

        # wait for data from the simulator
        log.msg('Waiting for data from the simulator.')
        from twisted.internet import reactor
        reactor.listenTCP(simOutputRecvPort, clientFactory, interface='')

        return clientFactory.deferred

    def set_data(self, data):
        self.data = data
        return data