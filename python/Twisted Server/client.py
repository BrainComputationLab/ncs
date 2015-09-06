import sys, thread
from socket import *
import struct
protobuf_path = ('../../base/include/ncs/proto')
sys.path.append(protobuf_path)
import SimData_pb2

def handler(clientSocket, addr):

	# temporarily write the live data to a file for comparision
	file = open("received.txt", "w")

	# protobuf message instance
	sim_data = SimData_pb2.SimData();

	first_message = True
	byte_count = 0

	while True:
		data = clientSocket.recv(4096)
		if not data:
			break

		if first_message:

			# receive routing key
			first_message = False
			key_size = int(data[0:3]) + 3
			routing_key = data[3:key_size]

			print "ROUTING KEY: " + routing_key
		else:

			# deserialize the protocol buffer
			sim_data.ParseFromString(data)
			if data:
			    byte_count += len(data)

			# unpack the bytes into floats
			temp = data.split()
			bytes = temp[len(temp) - 1] 
			if len(bytes) < 4:
			    bytes = bytes.zfill(4)

			try:
			    value = struct.unpack('f', bytes)[0]
			except:
			    print "Deserialize error:", sys.exc_info()[0]

			file.write(str(value)+'\n')

	'''
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
	'''

	# close sockets
	print 'PYTHON BYTE COUNT: ' + str(byte_count)
	clientSocket.close()
	file.close()



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
	clientSocket.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
	clientSocket.bind((host, port))
	clientSocket.listen(5)

	while True:
		clientsock, addr = clientSocket.accept()
		thread.start_new_thread(handler, (clientsock, addr))
