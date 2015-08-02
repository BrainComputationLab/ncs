# -*- coding: utf-8 -*-

import optparse, os, sys, ncs, struct
from twisted.internet.defer import Deferred, succeed
from twisted.internet.protocol import ClientFactory, ServerFactory, Protocol, ClientCreator
from twisted.internet.defer import inlineCallbacks, returnValue
from twisted.internet import reactor, task
from twisted.application import internet, service
from twisted.python import log
from txamqp.protocol import AMQClient
from txamqp.client import TwistedDelegate
from txamqp.content import Content
import txamqp.spec

protobuf_path = ('../../base/include/ncs/proto')
sys.path.append(protobuf_path)
import SimData_pb2

DEBUG = True

# static port for receiving streamed data from NCS
simOutputRecvPort = 8005

class RecvDataProtocol(Protocol):

    def __init__(self):
        self.deferred = None
        self.testfile = None
        self.routing_key = None
        self.first_message = True

    @inlineCallbacks
    def gotConnection(self, connection, username, password):
        yield connection.authenticate(username, password)
        if DEBUG:
            print "Authenticated. Ready to send messages"

        channel = yield connection.channel(1)
        yield channel.channel_open()
        returnValue((connection, channel))

    @inlineCallbacks
    def send_message(self, result, message):
        connection, channel = result
        msg = Content(message)
        msg["delivery mode"] = 2

        yield channel.basic_publish(exchange='datastream', routing_key=self.routing_key, content=msg)

        if DEBUG:
            print "Sending message: %s" % message
        returnValue(result)

    @inlineCallbacks
    def cleanup(self, result):
        connection, channel = result
        stopToken = "STOP"
        msg = Content(stopToken)
        msg["delivery mode"] = 2
        channel.basic_publish(exchange='datastream', content=msg, routing_key=self.routing_key)
        yield channel.channel_close()
        chan0 = yield connection.channel(0)
        yield chan0.connection_close()

    def connectionMade(self):

        if DEBUG:
            self.testfile = open("testing-1-2-3.txt", "w")

        spec = txamqp.spec.load("txamqp/amqp0-8.stripped.rabbitmq.xml")
        delegate = TwistedDelegate()

        self.deferred = ClientCreator(reactor, AMQClient, delegate=delegate, vhost='/', spec=spec).connectTCP("localhost", 5672)
        self.deferred.addCallback(self.gotConnection, "guest", "guest")

    def dataReceived(self, data):

        if self.first_message:

            # receive routing key
            self.first_message = False
            key_size = int(data[0:3]) + 3
            self.routing_key = data[3:key_size]

            if DEBUG:
                print "ROUTING KEY: " + self.routing_key
        else:

            # deserialize the protocol buffer
            sim_data = SimData_pb2.SimData();
            sim_data.ParseFromString(data)

            # unpack the bytes into floats
            temp = data.split()
            bytes = temp[len(temp) - 1] 
            if len(bytes) < 4:
                bytes = bytes.zfill(4)

            value = struct.unpack('f', bytes)[0]

            if DEBUG:
                self.testfile.write(str(value)+'\n')
            self.deferred.addCallback(self.send_message, str(value))

    def connectionLost(self, reason):
        if DEBUG:
            self.testfile.close()
        self.deferred.addCallback(self.cleanup)

class RecvDataProtocolFactory(ServerFactory):

    protocol = RecvDataProtocol

    def __init__(self, service):
        self.service = service

class RecvDataService(service.Service):

    def startService(self):
        service.Service.startService(self)
        log.msg('Service for receiving simulation data has been started')

# Not executed when ran as a service in the daemon
if __name__ == "__main__":

    # listen for connections for receiving data from the simulator
    service = RecvDataService()
    recvFactory = RecvDataProtocolFactory(service)
    port = reactor.listenTCP(simOutputRecvPort, recvFactory)

    reactor.run()