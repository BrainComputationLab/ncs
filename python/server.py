#from __future__ import unicode_literals
from geventwebsocket.handler import WebSocketHandler
from gevent.pywsgi import WSGIServer
from flask import Flask, request, jsonify, send_from_directory
import time
import json

import thread
from socket import *
import os, gc, sys
from threading import Thread
import subprocess
import struct
import ncs
import array
import yaml

protobuf_path = ('../base/include/ncs/proto')
sys.path.append(protobuf_path)
import SimData_pb2
from model import ModelService

inputPort = 8004
outputPort = 8005

group_1 = None

class SimThread(Thread):
    """Thread that contains the running simulation."""

    def __init__(self, sim, max_steps, script):
        # call the superstructor for the Thread class, otherwise demons emerge
        super(SimThread, self).__init__()
        self.sim = sim
        self.max_steps = max_steps
        self.step = 0
        self.script = script

    def run(self):
    	# run the script that was passed in
		full_path = os.path.abspath(self.script)
		os.system("chmod +x " + full_path)
		#subprocess.call(full_path, shell=True)

		sim.run(duration=1.0)    
		print "Simulation Complete!"
		del self.sim

		# force the sim to be garbage collected
		gc.collect()

# test for running a sim with params from a JSON object
def jsonTest(sim):
	# declare data as a python dictionary
	sim_params = {
    	"top_group": "hf8hfh8h8dgf8shad0fhsd0fh",
    	"neurons": [
    		{
    			"_id": "ajsd9fd90ha0hsd80fhd80sha",
    			"name": "neuron_izh_1",
    			"description": "regular spiking neuron",
    			"author": "Nathan Jordan",
    			"specification": 
    			{
        			"neuron_type": "izhikevich",
        			"a": 0.02,
        			"b": 0.2,
        			"c": -65.0,
        			"d": 8.0,
        			"u": -12.0,
        			"v": -60.0,
        			"threshold": 30
    			}
			}
    	],
    	"synapses": [
    	],
    	#parameters for addStimulus function:
        #       1. A stimulus type (string) 
        #               rectangular_current, rectangular_voltage, linear_current, linear_voltage, 
        #               sine_current, or sine_voltage
        #       2. A map from parameter names (strings) to their values (Generators)
        #               Parameter names are amplitude, starting_amplitude, ending_amplitude,
        #               delay, current, amplitude_scale, time_scale, phase, amplitude_shift,
        #               etc. based on the stimulus type
        #       3. A set of target neuron groups
        #       4. probability of a neuron receiving input
        #       5. start time for stimulus (seconds)
        #               For example, if you wanted to start stimulus at 1 ms, write 0.01
        #       6. end time for stimulus (seconds)
    	"stimuli": [
			{
   	 			"_id": "df90sahf0sd9ha8sdhf8dhsa",
    			"entity_type": "stimulus",
    			"entity_name": "stimulus_1",
    			"description": "This is an extended description of the entity",
    			"author": "Nathan Jordan",
    			"author_email": "njordan@cse.unr.edu",
    			"specification":
    			{
        			"stimulus_type": "rectangular_current",
        			"amplitude": 10,

        			# where do we get these? Replace with neuron group?
        			"width": 3,
        			"frequency": 10,

        			"probability": 1.0,
        			"time_start": 0.0,
        			"time_end": 1.0
    			}
			}
    	],
    	#addReport function
        #Parameters for addReport function:
        #       1. A set of neuron group or a set of synapse group to report on
        #       2. A target type: "neuron" or "synapses"
        #       3. type of report: synaptic_current, neuron_voltage, neuron_fire, 
        #          input current, etc.
        #       4. Probability (the percentage of elements to report on)
    	"reports": [
			{
    			"_id": "df90sahf0sd9ha8sdhf8dhsa",
    			"entity_type": "report",
    			"entity_name": "report_1",
    			"description": "This is an extended description of the entity",
    			"author": "Nathan Jordan",
    			"author_email": "njordan@cse.unr.edu",
    			"specification": 
    			{
        			"report_type": "neuron_voltage",
        			"report_targets": [
            		"group_1"
        			],
        			"probability": 1.0,
        			"frequency": 5,
        			"channel_types": [
            		"9fh09dhf90asdhf9dsahf9hasd9f",
            		"gf8dshf80h08ah0shdf8h8ays89d"
        			],
        			"method": {
           	 			"type": "file",
            			"filename": "regular_spiking_izh_json_test.txt",
            			"number_format": "ascii"
       	 			},
        			"time_start": 0.0,
        			"time_end": 1.0
    			}
			}
    	],
    	"groups": [
    	],
    	"neuron_groups": [
			{
				"location_string": "group_1",
				"count": 1,
    			"neuron":
    				{
	    				"_id": "ajsd9fd90ha0hsd80fhd80sha",
	    				"name": "neuron_izh_1",
	    				"description": "regular spiking neuron",
	    				"author": "Nathan Jordan",
	    				"specification": 
	    				{
		        			"neuron_type": "izhikevich",
		        			"a": 0.02,
		        			"b": 0.2,
		        			"c": -65.0,
		        			"d": 8.0,
		        			"u": -12.0,
		        			"v": -60.0,
		        			"threshold": 30
	    				}
					}
			}
		]
	}

	#addNeuronGroup function
        #Parameters for addNeuronGroup function:
        #       1. A name of the group (string)
        #       2. Number of cells (integer)
        #       3. Neuron parameters
        #       4. Geometry generator (optional)
	#group_1=sim.addNeuronGroup("group_1",1,regular_spiking_parameters,None)
	neuron_group_dict = {
		"neuron_groups": [
			{
				"local_string": "group_1",
				"count": 1,
    			"neuron": [
    				{
	    				"_id": "ajsd9fd90ha0hsd80fhd80sha",
	    				"name": "neuron_izh_1",
	    				"description": "regular spiking neuron",
	    				"author": "Nathan Jordan",
	    				"specification": 
	    				{
		        			"neuron_type": "izhikevich",
		        			"a": 0.02,
		        			"b": 0.2,
		        			"c": -65.0,
		        			"d": 8.0,
		        			"u": -12.0,
		        			"v": -60.0,
		        			"threshold": 30
	    				}
					}
	    		]
			}
		]
	}

	# use dumps to convert dictionary to a json object
	jsonObj = json.dumps(sim_params)
	group_name = json.dumps(neuron_group_dict)

	temp = yaml.load(jsonObj)
	temp2 = yaml.load(group_name)
	#jsonObj = json.loads(sim_params)
	#group_name = json.loads(neuron_group_dict)

	# assigns sim params based on json properties
	modelService = ModelService()
	#sim = modelService.sim_test(sim);
	#return sim
	# this function takes dictionaries (converted json objects)
	# handles assigning neurons, synapses, and groups
	group_1 = modelService.process_model(sim, temp, temp2)	

	if not sim.init(sys.argv):
		print "failed to initialize simulation."
		return	

	modelService.add_stims_and_reports(sim, temp, temp2, group_1)

'''def acceptClientsForInput(inputServerSocket):

	# listen forever
	while True:
		# accept connections
		inputAcceptedSocket, addr = inputServerSocket.accept()
		thread.start_new_thread(handleInputSocket,(inputAcceptedSocket,))

	# close the sockets
	inputServerSocket.close()'''

def acceptClientsForOutput(outputServerSocket):

	# listen forever
	while True:
		# accept connections
		outputAcceptedSocket, addr = outputServerSocket.accept()
		thread.start_new_thread(handleOutputSocket,(outputAcceptedSocket,))

	# close the sockets
	outputServerSocket.close()	

#function for handling simulation input parameters
def handleInputSocket(clientSocket):
	print 'Established connection with NCB'

	# temporarily write the live data to a file for comparision
	file = open("json.txt", "w")
	
	#receive the simulation parameters
	input_data = clientSocket.recv(4096)
	#file.write(input_data + '\n\n\n')

	print 'loading JSON...'

	json_obj = json.loads(input_data)
	json_model = json_obj['model']
	json_sim_input_and_output = json_obj['simulation']
	#print json_obj

	# dumps is to encoded
	# loads is to decoded from JSON to python

	file.write(json.dumps(json_obj, sort_keys=True, indent=2) + '\n\n\n')

	file.write('MODEL:\n')
	file.write(json.dumps(json_model, sort_keys=True, indent=2) + '\n\n\n')
	file.write('SIMULATION:\n')
	file.write(json.dumps(json_sim_input_and_output, sort_keys=True, indent=2) + '\n')

	modelService = ModelService()
	#sim = modelService.sim_test(sim);
	#return sim
	# this function takes dictionaries (converted json objects)
	# handles assigning neurons, synapses, and groups
	neuron_groups = []
	synapse_groups = []
	sim = ncs.Simulation()
	modelService.process_model(sim, json_model, neuron_groups, synapse_groups)
	sim.init(sys.argv)
	#modelService.add_stims_and_reports(sim, json_sim_input_and_output, json_model, neuron_groups, synapse_groups)
	sim.run(duration=1.0)
	# close sockets
	file.close()
	clientSocket.close()	

#function for handling streaming simulation output
def handleOutputSocket(clientSocket):
	print 'Established connection with simulator'

	# create socket to stream simulation data
	outputSocket = socket(AF_INET, SOCK_STREAM)
	try:
		outputSocket.connect((host,8004))
		print 'Established connection with web server'
	except Exception, e:
		print >>sys.stderr, "Error with output socket. Exception type is " + str(e)

	# protobuf message instance
	sim_data = SimData_pb2.SimData();

	# temporarily write the live data to a file for comparision
	file = open("streamedData.txt", "w")

	# use this to only print the exception for the first attempt to send a data value
	flag = True
	
	while True:

		# receive the message length
		msg = (clientSocket.recvfrom(1))[0]
		if not msg:
			break
		msg_size = int(msg)	

		# receive the report name
		#report_name_size = int((connectionSocket.recvfrom(2))[0])
		#report_name = (connectionSocket.recvfrom(report_name_size))[0]
		#msg_size -= (report_name_size + 2)

		# ensure we received the entire message before deserializing
		buffer = ''
		while len(buffer) < msg_size:
			chunk = clientSocket.recv(msg_size - len(buffer))
			if chunk == '':
				break
			buffer += chunk

		# deserialize the protocol buffer
		sim_data.ParseFromString(buffer)
		if not buffer:
			break

		# for now we are just sending the simulation data to the web server
		# there is no way to determine which report this is currently
		# the web server will need to unpack the floating point data values

		# append the size of the buffer so the web server knows how much it needs to receive before unblocking
		buffer = str(len(buffer)) + buffer
		try:
			outputSocket.send(buffer)
		except Exception, e:
			if flag:
				print >>sys.stderr, "Error with output socket. Exception type is " + str(e)	
				flag = False

		# unpack the bytes into floats
		temp = buffer.split()
		bytes = temp[len(temp) - 1]	
		if len(bytes) < 4:
			bytes = bytes.zfill(4)
		value = struct.unpack('f', bytes)[0]

		# write the data to a file to check its correctness
		file.write(str(value)+'\n')	 

	# close sockets
	file.close()
	clientSocket.close()
	outputSocket.close()

# make sure run as main script
if __name__ == '__main__':

	#check for sim script command line argument
	if len(sys.argv) != 2:
		print 'Error: You must pass in a simulation script (e.g. python server.py samples/models/izh/regular_spiking_izh.py)'
		sys.exit(1)
	else:
		sim_script = sys.argv[1]

	# for now all socket communication will use the local host	
	host = gethostbyname(gethostname())	

	# create TCP socket to receive simulation input parameters
	inputSocket = socket(AF_INET, SOCK_STREAM)
	inputSocket.bind((host,inputPort))
	
	# init queue to 5 sockets
	inputSocket.listen(5)
	print 'Ready for input at address:', inputSocket.getsockname()[0], 'on port:', inputPort

	# create TCP socket to receive simulation output data
	outputSocket = socket(AF_INET, SOCK_STREAM)
	outputSocket.bind((host,outputPort))
	
	# init queue to 5 sockets
	outputSocket.listen(5)
	print 'Ready for output at address:', outputSocket.getsockname()[0], 'on port:', outputPort

	# wait for streaming data on a separate thread
	#thread.start_new_thread(acceptClientsForInput,(inputSocket,))
	thread.start_new_thread(acceptClientsForOutput,(outputSocket,))

	# start running the simulation
	
	#jsonTest(sim)
	#sim_thread = SimThread(sim, 5, sim_script)
	#sim_thread.start()

	# listen forever
	while True:
		# accept connections
		inputAcceptedSocket, addr = inputSocket.accept()
		thread.start_new_thread(handleInputSocket,(inputAcceptedSocket,))

	# close the sockets
	inputSocket.close()

