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
	          "$$hashKey": "09B", 
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
		2.Add new user
		3.Save a model
		4.Update models
		5.Quit
		""")

		ans=raw_input("What would you like to do? ")
		if ans=="1":
			launch_sim(clientSocket)  
		elif ans=="2":
			pass
		elif ans=="3":
			pass
		elif ans=="4":
			pass
		elif ans=="5":
			ans = None
		else:
			print("\n Not Valid Choice. Try again")

	# close socket
	clientSocket.close()