# -*- coding: utf-8 -*-

'''
This is the service that runs as a daemon process
It is responsible for running the server and logging

Run this with 
twistd --python server_daemon.py
or twistd --nodaemon --python server_daemon.py
to run it as a foreground process

if ran as a foreground process, the logging will 
output to the terminal instead of a file

kill $(cat twistd.pid) 
to stop it if it is running as a background process
'''

from twisted.application import internet, service
from twisted.internet.protocol import ServerFactory, Protocol
from twisted.python import log

from stream_data_proxy import ProxyService, RecvSimDataProxyProtocolFactory
from add_user import AddUserService, AddUserProtocolFactory
from authenticator import AuthenticationService, AuthenticationServiceFactory

# configuration parameters
sim_port = 10000
add_user_port = 8009
ncb_port = 8005
iface = '127.0.1.1'

# this will hold the services that combine to form the server
top_service = service.MultiService()

# service for receiving sim output and relaying it to clients
proxy_service = ProxyService()
proxy_service.setServiceParent(top_service)
factory = RecvSimDataProxyProtocolFactory(proxy_service)
tcp_service = internet.TCPServer(sim_port, factory, interface=iface)
tcp_service.setServiceParent(top_service)

# service that handles request from NCB to add a new user
add_user_service = AddUserService()
add_user_service.setServiceParent(top_service)
add_user_factory = AddUserProtocolFactory(add_user_service)
tcp_service = internet.TCPServer(add_user_port, add_user_factory, interface=iface)
tcp_service.setServiceParent(top_service)

# service that handles requests from NCB users
auth_service = AuthenticationService()
auth_service.setServiceParent(top_service)
auth_factory = AuthenticationServiceFactory(auth_service)
auth_tcp_service = internet.TCPServer(ncb_port, auth_factory, interface=iface)
auth_tcp_service.setServiceParent(top_service)

# this creates an application and hooks the service collection to it
application = service.Application("daemon")
top_service.setServiceParent(application)

# print address and port incase logging goes to a file
print 'Serving on %s, port %d'  % (iface, ncb_port)
