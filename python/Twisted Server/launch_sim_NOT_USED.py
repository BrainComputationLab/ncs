# -*- coding: utf-8 -*-

import optparse, os

from twisted.internet.defer import Deferred, succeed
from twisted.internet.protocol import ClientFactory, ServerFactory, Protocol
from twisted.application import internet, service
from twisted.python import log
import json
import sys, ncs

from model import ModelService

# This factory and service are used for receiving simulation parameters
class RecvSimParamsProxyProtocol(Protocol):

    # IS THIS MEMBER NEEDED?
    params = ''

    # This is a built in function that is overrided here. It is invoked whenever a message is received
    # WILL THIS RECEIVE THE ENTIRE MESSAGE (STREAM?)
    def dataReceived(self, params):
        log.msg('Established connection with NCB')
        deferred = self.factory.service.recv_params(params)

        # send the client the data when we receive it
        #deferred.addCallback(self.transport.write)
        # THIS IS WHERE WE WANT TO RUN THE SIM

        # this closes the connection (once the callbacks are finished)
        # we'll leave this here until we change it so the proxy sends data
        # as it gets it (so it shouldn't close the connection)
        #deferred.addBoth(lambda r: self.transport.loseConnection())
        self.transport.loseConnection() # WE DONT WANT THIS!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
        #self.totalData += data

    # this depends on the fact that the sim closes the connection when it is done sending
    def connectionLost(self, reason):
        self.fullStreamReceived(self.params)

    def fullStreamReceived(self, totalData):
        #self.factory.stream_finished(totalData)
        pass

class RecvSimParamsProxyProtocolFactory(ServerFactory):

    protocol = RecvSimParamsProxyProtocol

    def __init__(self, service):
        self.service = service

class SimParamsProxyService(service.Service):

    data = None

    def __init__(self):
        # set recv factory instance?
        pass

    def startService(self):
        service.Service.startService(self)
        log.msg('Service for receiving sim parameters is running')

    def recv_params(self, params):

        # temporarily write the JSON to a file for comparision
        file = open("json_recvd.txt", "w")

        json_obj = json.loads(params)
        json_model = json_obj['model']
        json_sim_input_and_output = json_obj['simulation']

        # dumps is to encoded
        # loads is to decoded from JSON to python

        file.write(json.dumps(json_obj, sort_keys=True, indent=2) + '\n\n\n')
        file.write('MODEL:\n')
        file.write(json.dumps(json_model, sort_keys=True, indent=2) + '\n\n\n')
        file.write('SIMULATION:\n')
        file.write(json.dumps(json_sim_input_and_output, sort_keys=True, indent=2) + '\n')
        file.close()

        # SHOULD THIS BE A TWISTED SERVICE NOW?
        modelService = ModelService()

        # this function takes dictionaries (converted json objects) and handles assigning neurons, synapses, and groups
        neuron_groups = []
        synapse_groups = []
        sim = ncs.Simulation()
        modelService.process_model(sim, json_model, neuron_groups, synapse_groups)
        print "ATTEMPTING TO INIT SIM..."
        if not sim.init(sys.argv):
            print "failed to initialize simulation." # THIS SHOULD BE AN ERROR CALLBACK
            return  
        print "ATTEMPTING TO ADD STIMS AND REPORTS"
        modelService.add_stims_and_reports(sim, json_sim_input_and_output, json_model, neuron_groups, synapse_groups)
