import sys
from socket import *
import json

def add_user(socket, already_exists):

	if (already_exists):
		params = {
			"request": "addUser",
			"user": {
	            "username": "testuser@gmail.com",
	            "password": "supersecretpassword",
	            "first_name": "Hersheys",
	            "last_name": "Bar",
	            "institution": "UNR",
	            "lab_id": 8675309,
	            "salt": None,
	            "models": None
			}
		}
	else:
		params = {
			"request": "addUser",
			"user": {
	            "username": "testuser2@gmail.com",
	            "password": "password2",
	            "first_name": "Bundt",
	            "last_name": "Cake",
	            "institution": "UNR",
	            "lab_id": 8675309,
	            "salt": None,
	            "models": None
			}
		}

	message = json.dumps(params)
	socket.send(message)


if __name__ == '__main__':

	# static domain for daemon add user service
	host = '127.0.1.1'
	port = 8009
		
	clientSocket = socket(AF_INET, SOCK_STREAM)
	clientSocket.connect((host,port))

	add_user(clientSocket, True)	 

	while True:
		data = clientSocket.recv(1024)
		if data:
			break
	print data

	# close socket
	clientSocket.close()