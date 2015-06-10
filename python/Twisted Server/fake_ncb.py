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
	    "duration": 0, 
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
	response = socket.recv(512)
	print response	


def save_model(socket, location):

	if location == 'personal':
		params = {
			"request": "saveModel",
		    "location": "blah",
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

# make sure run as main script
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

	# ADD PROMPT FOR USERNAME/PASSWORD

	# send [serialized] credentials
	credentials = json.dumps({"request": "login", "username": "testuser@gmail.com", "password": "supersecretpassword"})
	clientSocket.send(credentials)
	data = clientSocket.recv(512)
	print data	 

	# menu for testing all possible commands
	ans = True
	while ans:
		print("""
		1.Launch a sim
		2.Save a model
		3.Undo model change
		4.Get current models
		5.Quit
		""")

		ans=raw_input("What would you like to do? ")
		if ans=="1":
			launch_sim(clientSocket)  
		elif ans=="2":
			save_model(clientSocket, 'personal')
		elif ans=="3":
			undo_model_change(clientSocket)
		elif ans=="4":
			get_models(clientSocket)
		elif ans=="5":
			ans = None
		else:
			print("\n Not Valid Choice. Try again")

	# close socket
	clientSocket.close()