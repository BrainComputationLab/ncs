import sys
from socket import *
import struct
protobuf_path = ('../../base/include/ncs/proto')
sys.path.append(protobuf_path)
import SimData_pb2

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

	# temporarily write the live data to a file for comparision
	file = open("received.txt", "w")
		
	#create socket
	clientSocket = socket(AF_INET, SOCK_STREAM)
	clientSocket.connect((host,port))

	# protobuf message instance
	sim_data = SimData_pb2.SimData();
	
	while True:

		# receive the message length
		msg = (clientSocket.recvfrom(1))[0]
		if not msg:
			break
		msg_size = int(msg)	

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

		# unpack the bytes into floats
		temp = buffer.split()
		bytes = temp[len(temp) - 1]	
		if len(bytes) < 4:
			bytes = bytes.zfill(4)
		value = struct.unpack('f', bytes)[0]

		# write the data to a file to check its correctness
		file.write(str(value)+'\n')	 

	# close sockets
	clientSocket.close()
	file.close()
