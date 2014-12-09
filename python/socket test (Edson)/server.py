
from __future__ import unicode_literals

from geventwebsocket.handler import WebSocketHandler
from gevent.pywsgi import WSGIServer

from flask import Flask, request, jsonify, send_from_directory

import time

import json

from socket import *
import struct

# Create new application
app = Flask(__name__)

# Debugging is okay for now
app.debug = True

@app.route('/')
def index_route():
    return app.send_static_file('index.html')




reports = []
fileIn = open('reg_voltage_report.txt', 'r')

@app.route('/report-<slug>')
def api(slug):

    if request.environ.get('wsgi.websocket'):
        ws = request.environ['wsgi.websocket']
        reports.append({"number": int(slug), "socket": ws})

        message = ws.receive()
        oldTime = time.time()

        #create TCP socket
        dataSocket = socket(AF_INET, SOCK_STREAM)
        host = gethostbyname(gethostname())
        port = 8005
        dataSocket.bind((host,port))
        dataSocket.listen(5)
        print 'Web Server ready at address:', dataSocket.getsockname()[0], 'on port:', port

        # wait until the connection is established
        connectionSocket, addr = dataSocket.accept()
        count = 0;
        while True:

            # receive the message length
            sizeStr = (connectionSocket.recvfrom(1))[0]
            if not sizeStr:
                break
            size = int(sizeStr) 
            count += 1
            print 'Msg Number: ' + str(count) + ' size: ' + str(size)
            # receive the data
            buffer = (connectionSocket.recvfrom(size))[0]
            #buffer = connectionSocket.recv(1024)
            if not buffer:
                break

            # unpack the bytes into floats
            temp = buffer.split()
            if len(temp) == 1:
                bytes = temp[0]
            else:
                bytes = temp[1] 
            if len(bytes) < 4:
                bytes = bytes.zfill(4)
            value = struct.unpack('f', bytes)[0]

            # Read from file
            #firstline = fileIn.readline().split()

            # Wait 1 second between sends
            #difference = 0
            #while difference < 1:
            #    difference = time.time() - oldTime
            #oldTime = time.time()

            # Do for each report
            for report in reports:
                number = report["number"]


                #value = float(firstline[number])
                #print value

                # Send in JSON format
                report["socket"].send(json.dumps(value))

        dataSocket.close()
        print 'Data Socket closed'

    return


# Serves static resources like index.html, css, js, images, etc.
@app.route('/assets/<path:resource>')
def serve_static_resource(resource):
    # Return the static file
    return send_from_directory('static/assets/', resource)


if __name__ == '__main__':
    http_server = WSGIServer(('localhost',8000), app, handler_class=WebSocketHandler)
    http_server.serve_forever()
