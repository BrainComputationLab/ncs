from zope.interface import implements, Interface
from twisted.cred import checkers, credentials, portal
from twisted.application import internet, service
from twisted.internet import protocol, reactor
from twisted.internet.protocol import ClientFactory, ServerFactory, Protocol
from twisted.protocols import basic
from twisted.python import log
import json

from simulation import Simulation
from credentials_checker import DBCredentialsChecker
import sys
sys.path.append('txmongo/ncs_db')
from database import TxMongoDatabase


DEBUG = True

# avatars are the objects (with associated functions) available for a particular user
class IProtocolAvatar(Interface): 

	def logout():
		"""Clean up per-login resources allocated to this avatar."""

class LoginAvatar(object): 
	implements(IProtocolAvatar)

	def logout(self): 
		pass

# the realm is the object that provides access to all available avatars
class Realm(object): 
	implements(portal.IRealm)

	# this method is invoked by the portal when a user successfully logs in
	def requestAvatar(self, avatarId, mind, *interfaces): 
		if DEBUG:
			print 'In requestAvatar with avatarId: ' + avatarId

		avatar = LoginAvatar()
		return IProtocolAvatar, avatar, avatar.logout

class AuthenicationServiceProtocol(Protocol):
	portal = None
	avatar = None
	logout = None
	service = None

	# a callback that is invoked after a client opens a connection
	def connectionMade(self):
		if DEBUG:
			print 'In connectionMade'

	def connectionLost(self, reason): 
		if DEBUG:
			print 'Connection lost due to: ' + str(reason)

		if self.logout:
			self.logout()
			self.avatar = None
			self.logout = None

	# called when a message is received from the client
	def dataReceived(self, data):

		# deserialize the data
		message = json.loads(data)

		if "request" in message:
			if message.get("request") in self.service.options:

				# User has not been granted an avatar (logging in)
				if not self.avatar and message.get("request") == "login":
					username = message.get('username')
					password = message.get('password')

					if DEBUG:
						print 'Received username: ' + username
						print 'Received password: ' + password

					self.tryLogin(username, password) 

				# execute request	
				else:
					if DEBUG:
						print 'Performing request: ' + message.get("request")

 					self.service.options[message.get("request")](message)

			else:
				self.transport.write('Invalid request. Goodbye.')
				self.transport.loseConnection()
		else:
			self.transport.write('Invalid request format. Goodbye.')
			self.transport.loseConnection()

	def tryLogin(self, username, password): 
		if DEBUG:
			print 'In tryLogin'

		#self.portal.login(credentials.UsernameHashedPassword(username, hash(password)), IProtocolAvatar).addCallbacks(self._cbLogin, self._ebLogin)
		self.portal.login(credentials.UsernameHashedPassword(username, password), IProtocolAvatar).addCallbacks(self._cbLogin, self._ebLogin)

	def _cbLogin(self, (interface, avatar, logout)): 
		if DEBUG:
			print 'In login callback'

		self.avatar = avatar
		self.logout = logout

		self.transport.write("Login successful, please proceed.")
		if DEBUG:
			print "Login successful, please proceed."

		# SEND MODELS AND ANY SIM OUTPUT DATA

	def _ebLogin(self, failure): 
		if DEBUG:
			print 'In login errorback'
			print 'Login denied, goodbye.'

		self.transport.loseConnection()

class AuthenticationServiceFactory(ServerFactory):

    protocol = AuthenicationServiceProtocol

    def __init__(self, service):
        self.service = service
        self.protocol.service = self.service
        self.protocol.portal = self.service.portal

class AuthenticationService(service.Service):

	def __init__(self):
		self.realm = Realm()
		self.portal = portal.Portal(self.realm)
		self.db = TxMongoDatabase("ncb")
		self.checker = DBCredentialsChecker(self.db)
		self.portal.registerChecker(self.checker)

		self.options = {
		        "login" : None,
		        "addUser": self.add_user,
				"launchSim" : self.launch_sim,
				"saveModel" : self.save_model,
				"updateModels": self.update_models
			}

	def startService(self):
		service.Service.startService(self)
		log.msg('Authentication service has been started')

	def add_user(self, params):
		pass

	def launch_sim(self, params):
		sim = Simulation()
		sim.build_sim(params)
		sim.run_sim(params)

	def save_model(self, params):
		pass

	def update_models(self, params):
		pass