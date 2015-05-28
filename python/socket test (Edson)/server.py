from __future__ import unicode_literals, print_function
from flask import Flask, request, jsonify, send_from_directory
import json, os

from socket import *
import struct
from geventwebsocket.handler import WebSocketHandler
from gevent.pywsgi import WSGIServer
import time


# Create new application
app = Flask(__name__, static_url_path='', static_folder='')
# Debugging is okay for now
app.debug = True

# set upload folder and allowed extensions
allowedFileExtensions = set(['json','py'])

# register upload folder with flask app
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['EXPORT_FOLDER'] = 'exports'

importFile = 'import.json'
importFilePath = os.path.join(app.config['UPLOAD_FOLDER'], importFile)
# changes old key names to new key names
def changeAllKeys(obj, oldKey, newKey):
    if not isinstance(obj, dict):
        return

    for key in obj:
        if key == oldKey:
            obj[newKey] = obj.pop(oldKey)
            key = newKey

        if isinstance(obj[key], dict):
            changeAllKeys(obj[key], oldKey, newKey)

        elif isinstance(obj[key], list):
            for element in obj[key]:
                changeAllKeys(element, oldKey, newKey)

# load json file
def loadJSONFile(fileName):
    try:
        with open(fileName) as fin:
            jsonObj = json.load(fin)

    except IOError:
        print("Error: %s not found." % (fileName,))
        return {'success': False}

    changeAllKeys(jsonObj, u'groups', u'cellGroups')
    changeAllKeys(jsonObj, u'neuron_aliases', u'cellAliases')
    changeAllKeys(jsonObj, u'entity_name', u'name')
    changeAllKeys(jsonObj, u'entity_type', u'type')

    return jsonObj

# saves json to file
def saveJSONFile(fileName, JSON):
    changeAllKeys(JSON, u'baseCellGroups', u'groups')
    changeAllKeys(JSON, u'cellGroups', u'groups')
    changeAllKeys(JSON, u'cellAliases', u'neuron_aliases')
    changeAllKeys(JSON, u'name', u'entity_name')
    changeAllKeys(JSON, u'type', u'entity_type')

    with open(fileName, 'w') as fout:
        json.dump(JSON, fout, indent=4)


#initiates transfer to ncs
@app.route('/transfer', methods=['POST', 'GET'])
def transferData():
    if request.method == 'POST':
        # jsonObj now has simulation parameters and model
        jsonObj = request.get_json(False, False, False)

        # send jsonObj to NCS here
        #print ("jsonObj: %r" %(jsonObj))
        return jsonify({'success': True})

    return jsonify({'success': False})


# initiates export
@app.route('/export', methods=['POST', 'GET'])
def exportFile():
    global exportFile
    if request.method == 'POST':
        jsonObj = request.get_json(False, False, False)
        fileName = jsonObj['name'] + '.json'
        filePath = os.path.join(app.config['EXPORT_FOLDER'], fileName)
        saveJSONFile(filePath, jsonObj)

        exportFile = fileName
        return send_from_directory(app.config['EXPORT_FOLDER'], fileName, as_attachment = True)
    elif request.method == 'GET':
        return send_from_directory(app.config['EXPORT_FOLDER'], exportFile, as_attachment = True)


    return jsonify({"success": False})

@app.route('/import', methods=['POST', 'GET'])
def importFile():
    if request.method == 'POST':
        #print ("files: %r" %(request.files))
        webFile = request.files['import-file'];
        # if file exists and is allowed extension
        if webFile: #and allowed_file(webFile.filename):
            # save file to server filesystem
            #name = secure_filename(importFile)
            #print(name)
            #print ("webfile: %r" %(webFile))
            webFile.save(importFilePath)
            #print("Here2")
            jsonObj = loadJSONFile(importFilePath)
            # return JSON object to determine success
            #print("GOT: ", end="")
            #print(jsonObj)
            return jsonify(jsonObj)

    elif request.method == 'GET':
        jsonObj = loadJSONFile(importFile)
        return jsonify(jsonObj)

    else:
        return jsonify({'success': False})

@app.route('/login')
def login_route():
    return 'not implemented', 500


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
        #print 'Web Server ready at address:', dataSocket.getsockname()[0], 'on port:', port

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
            #print 'Msg Number: ' + str(count) + ' size: ' + str(size)
            # receive the data
            buffer = (connectionSocket.recvfrom(size))[0]

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

            '''# Read from file
            firstline = fileIn.readline().split()

            # Wait 1 second between sends
            difference = 0
            while difference < 1:
                difference = time.time() - oldTime
            oldTime = time.time()'''

            # Do for each report
            for report in reports:
                number = report["number"]


                #value = float(firstline[number])
                #print value

                # Send in JSON format
                report["socket"].send(json.dumps(value))

        dataSocket.close()
        print ('Data Socket closed')

    return


# Serves static resources like index.html, css, js, images, etc.
@app.route('/assets/<path:resource>')
def serve_static_resource(resource):
    # Return the static file
    return send_from_directory('static/assets/', resource)


# If we're running this script directly (eg. 'python server.py')
# run the Flask application to start accepting connections
if __name__ == "__main__":
    app.run('localhost', 8007)
