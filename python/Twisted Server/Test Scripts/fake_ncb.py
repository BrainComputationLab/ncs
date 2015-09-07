import sys
from socket import *
import json

def launch_sim(socket):
	params = {
	"request": "launchSim",
	"model":
	{
	  "author": "", 
	  "cellAliases": [], 
	  "cellGroups": {
	    "cellGroups": [
	      {
	        "$$hashKey": "052", 
	        "classification": "cells", 
	        "description": "Description", 
	        "geometry": "Box", 
	        "name": "Cell 1", 
	        "num": 100, 
	        "parameters": {
	          "calcium": {
	            "maxValue": 0, 
	            "minValue": 0, 
	            "type": "exact", 
	            "value": -65
	          }, 
	          "calciumSpikeIncrement": {
	            "maxValue": 0, 
	            "minValue": 0, 
	            "type": "exact", 
	            "value": 8
	          }, 
	          "channel": [], 
	          "leakReversalPotential": {
	            "maxValue": -55, 
	            "minValue": -75, 
	            "type": "uniform", 
	            "value": 0
	          }, 
	          "rMembrane": {
	            "maxValue": 0, 
	            "minValue": 0, 
	            "type": "exact", 
	            "value": 30
	          }, 
	          "restingPotential": {
	            "maxValue": 0, 
	            "minValue": 0, 
	            "type": "exact", 
	            "value": 0.2
	          }, 
	          "spikeShape": {
	            "maxValue": 0, 
	            "minValue": 0, 
	            "type": "exact", 
	            "value": 30
	          }, 
	          "tauCalcium": {
	            "maxValue": -11, 
	            "minValue": -15, 
	            "type": "uniform", 
	            "value": 0
	          }, 
	          "tauMembrane": {
	            "maxValue": 0, 
	            "minValue": 0, 
	            "type": "exact", 
	            "value": 30
	          }, 
	          "threshold": {
	            "maxValue": 0, 
	            "minValue": 0, 
	            "type": "exact", 
	            "value": 0.2
	          }, 
	          "type": "NCS"
	        }
	      }, 
	      {
	        "$$hashKey": "05A", 
	        "classification": "cells", 
	        "description": "Description", 
	        "geometry": "Sphere", 
	        "name": "Cell 3", 
	        "num": 150, 
	        "parameters": {
	          "a": {
	            "maxValue": 0, 
	            "minValue": 0, 
	            "type": "exact", 
	            "value": 0.2
	          }, 
	          "b": {
	            "maxValue": 0, 
	            "minValue": 0, 
	            "type": "exact", 
	            "value": 0.2
	          }, 
	          "c": {
	            "maxValue": 0, 
	            "minValue": 0, 
	            "type": "exact", 
	            "value": -65
	          }, 
	          "d": {
	            "maxValue": 0, 
	            "minValue": 0, 
	            "type": "exact", 
	            "value": 8
	          }, 
	          "threshold": {
	            "maxValue": 0, 
	            "minValue": 0, 
	            "type": "exact", 
	            "value": 30
	          }, 
	          "type": "Izhikevich", 
	          "u": {
	            "maxValue": -11, 
	            "minValue": -15, 
	            "type": "uniform", 
	            "value": 0
	          }, 
	          "v": {
	            "maxValue": -55, 
	            "minValue": -75, 
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
	  "name": "Current Model", 
	  "synapses": [
	    {
	      "$$hashKey": "05Z", 
	      "classification": "synapseGroup", 
	      "description": "Description", 
	      "parameters": {
	        "aLtdMinimum": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "aLtpMinimum": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "delay": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "lastPostfireTime": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "lastPrefireTime": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "maxConductance": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "name": "ncsSynapse", 
	        "psgWaveformDuration": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "redistribution": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "reversalPotential": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "tauDepression": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "tauFacilitation": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "tauLtd": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "tauLtp": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "tauPostSynapticConductance": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }, 
	        "utilization": {
	          "maxValue": 0, 
	          "minValue": 0, 
	          "type": "exact", 
	          "value": 30
	        }
	      }, 
	      "post": "Cell 3", 
	      "postPath": [
	        {
	          "$$hashKey": "05X", 
	          "index": 0, 
	          "name": "Home"
	        }
	      ], 
	      "pre": "Cell 1", 
	      "prePath": [
	        {
	          "$$hashKey": "05V", 
	          "index": 0, 
	          "name": "Home"
	        }
	      ], 
	      "prob": 0.5
	    }
	  ]
	},
	"simulation":
	{
	  "duration": 1, 
	  "fsv": None, 
	  "includeDistance": "No", 
	  "inputs": [
	    {
	      "$$hashKey": "061", 
	      "amplitude": 2, 
	      "className": "simulationInput", 
	      "endTime": 1000000, 
	      "frequency": 10, 
	      "inputTarget": "No Cell Groups Available", 
	      "name": "Input1", 
	      "probability": 0.5, 
	      "startTime": 500000, 
	      "stimulusType": "Rectangular Current", 
	      "width": 3
	    }
	  ], 
	  "interactive": "No", 
	  "name": "Sim", 
	  "outputs": [
	    {
	      "$$hashKey": "064", 
	      "className": "simulationOutput", 
	      "endTime": 1, 
	      "fileName": "output", 
	      "frequency": 10, 
	      "name": "Output1", 
	      "numberFormat": "ascii", 
	      "outputType": "Save As File", 
	      "saveAsFile": True,
	      "probability": 0.5, 
	      "reportTarget": "target", 
	      "reportType": "neuron_voltage", 
	      "startTime": 0
	    }
	  ], 
	  "seed": None
	}
	}

	message = json.dumps(params)
	socket.send(message)
	response = socket.recv(512)
	print response	


def save_model(socket, location):

	# Yes, this is gross, but it shows the supported locations
	if location == 'personal':
		params = {
			"request": "saveModel",
		    "location": "personal",
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
            "description": "NEWER MODEL", 
            "name": "Test Model", 
            "synapses": []
          }
		}

	elif location == 'lab':
		params = {
			"request": "saveModel",
		    "location": "lab",
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
		
	elif location == 'global':
		params = {
			"request": "saveModel",
			"location": "global",
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
		    "name": "Current Model", 
		    "synapses": []
		  }
		}

	else:
		print 'Invalid location'
		return

	message = json.dumps(params)
	socket.send(message)
	response = socket.recv(512)
	print response	

def undo_model_change(socket):
	params = {
			"request": "undoModelChange",
		    "location": "personal",
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
            "description": "This should work now", 
            "name": "Test Model", 
            "synapses": []
          }
		}

	message = json.dumps(params)
	socket.send(message)
	data = socket.recv(4096)
	print data

def get_models(socket):
	params = {
		"request": "getModels"
	}

	message = json.dumps(params)
	socket.send(message)
	data = socket.recv(4096)
	print data

def export_to_script(socket):
	params = {
		"request": "exportScript",
	  	"model": 
	  	{
	    "author": "", 
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
	    "name": "Current Model", 
	    "synapses": []
	  }, 
	  "simulation": {
	    "duration": 1, 
	    "fsv": None, 
	    "includeDistance": "No", 
	    "inputs": [], 
	    "interactive": "No", 
	    "name": None, 
	    "outputs": [], 
	    "seed": None
	  }
	}

	message = json.dumps(params)
	socket.send(message)
	data = socket.recv(4096)
	print data

def script_to_json(socket):
	params = {
		"request": "scriptToJSON"
	}

	message = json.dumps(params)
	socket.send(message)
	data = socket.recv(4096)

	print data

def logout(socket):
	params = {
		"request": "logout"
	}

	message = json.dumps(params)
	socket.send(message)
	data = socket.recv(4096)
	print data

if __name__ == '__main__':

	#check for correct input
	if len(sys.argv) != 3:
		print 'Usage: python client.py <server address> <server port>'
		sys.exit(1)
	
	#if correct input, set variables
	else:
		host = sys.argv[1]
		port = int(sys.argv[2])
		
	#create socket
	clientSocket = socket(AF_INET, SOCK_STREAM)
	clientSocket.connect((host,port))

	# send [serialized] credentials
	credentials = json.dumps({"request": "login", "username": "testuser2@gmail.com", "password": "password2"})
	clientSocket.send(credentials)
	data = clientSocket.recv(512)
	print data	 
	data = json.loads(data)
	if data['response'] == 'failure':
		sys.exit(1)

	# menu for testing all possible commands
	ans = True
	while ans:
		print("""
		1.Launch a sim
		2.Save a model
		3.Undo model change
		4.Get current models
		5.Export to Python script
		6.Convert Python script to JSON 
		7.Logout 
		8.Quit
		""")

		ans=raw_input("Request: ")
		if ans=="1":
			launch_sim(clientSocket)  
		elif ans=="2":
			save_model(clientSocket, 'personal')
		elif ans=="3":
			undo_model_change(clientSocket)
		elif ans=="4":
			get_models(clientSocket)
		elif ans=="5":
			export_to_script(clientSocket)
		elif ans=="6":
			script_to_json(clientSocket)
		elif ans=="7":
			logout(clientSocket)
		elif ans=="8":
			ans = None
		else:
			print("\n Invalid Choice.")

	# close socket
	clientSocket.close()