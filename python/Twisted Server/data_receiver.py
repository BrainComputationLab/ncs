# -*- coding: utf-8 -*-

import optparse, os
from twisted.internet.defer import Deferred, succeed
from twisted.internet.protocol import ClientFactory, ServerFactory, Protocol
from twisted.application import internet, service
from twisted.python import log
import sys, ncs, struct

protobuf_path = ('../../base/include/ncs/proto')
sys.path.append(protobuf_path)
import SimData_pb2

DEBUG = True

# static port for receiving streamed data from NCS
simOutputRecvPort = 8005

class RecvDataProtocol(Protocol):

    def __init__(self):
        self.fifo = None
        self.testfile = open("testfile", "w")

    def dataReceived(self, data):

        print 'IN DATA RECV'

        if self.fifo:

            # protobuf message instance
            sim_data = SimData_pb2.SimData();

            # receive the message length
            #msg = (clientSocket.recvfrom(1))[0]
            #msg_size = int(msg) 

            # receive the report name
            #report_name_size = int((connectionSocket.recvfrom(2))[0])
            #report_name = (connectionSocket.recvfrom(report_name_size))[0]
            #msg_size -= (report_name_size + 2)

            # ensure we received the entire message before deserializing
            '''buffer = ''
            while len(buffer) < msg_size:
                chunk = clientSocket.recv(msg_size - len(buffer))
                if chunk == '':
                    break
                buffer += chunk'''

            # deserialize the protocol buffer
            sim_data.ParseFromString(data)

            # unpack the bytes into floats
            temp = data.split()
            bytes = temp[len(temp) - 1] 
            if len(bytes) < 4:
                bytes = bytes.zfill(4)
            value = struct.unpack('f', bytes)[0]
            self.testfile.write(str(value)+'\n')
            self.fifo.write(str(value)+'\n')

        else:
            # invalid first message (does not contain username and report name)
            if '.' not in data: # IS THIS HOW WE PLAN TO SEND IT?
                self.transport.loseConnection()
            else:
                #path = os.path.join()
                path = 'data/' + data.split('.')[0] + data.split('.')[1]

                if DEBUG:
                    print 'Fifo path: ' + path

                os.mkfifo(path)
                self.fifo = open(path, 'w')

    def connectionLost(self, reason):
        if self.fifo is not None:
            self.fifo.close()
        self.testfile.close()

class RecvDataProtocolFactory(ServerFactory):

    protocol = RecvDataProtocol

    def __init__(self, service):
        self.service = service

class RecvDataService(service.Service):

    data = None

    def __init__(self):
        # set recv factory instance?
        pass

    def startService(self):
        service.Service.startService(self)
        log.msg('Service for receiving simulation data has been started')