from zope.interface import implements, Interface
from twisted.cred import checkers, credentials, portal
from twisted.application import internet, service
from twisted.internet.defer import Deferred, maybeDeferred
from twisted.internet import protocol, reactor, defer
from twisted.internet.protocol import ClientFactory, ServerFactory, Protocol
from twisted.protocols import basic
from twisted.python import log
import json, time

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

	avatar_id = None

	# this method is invoked by the portal when a user successfully logs in
	def requestAvatar(self, avatarId, mind, *interfaces): 
		if DEBUG:
			print 'In requestAvatar with avatarId: ' + avatarId

		self.avatar_id = avatarId
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
					self.service.username = message.get('username')
					password = message.get('password')

					if DEBUG:
						print 'Received username: ' + self.service.username
						print 'Received password: ' + password

					self.tryLogin(self.service.username, password) 

				# execute request	
				else:
					if DEBUG:
						print 'Performing request: ' + message.get("request")

 					deferred = self.service.options[message.get("request")](message)

 					# notify client of success or failure
 					deferred.addCallbacks(self.request_done, self.request_done)

			else:
				self.transport.write('Invalid request. Goodbye.')
				self.transport.loseConnection()
		else:
			self.transport.write('Invalid request format. Goodbye.')
			self.transport.loseConnection()

	def request_done(self, ign):
		self.transport.write(self.service.response)

	def tryLogin(self, username, password): 
		if DEBUG:
			print 'In tryLogin'

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

	username = None
	response = None

	def __init__(self):
		self.realm = Realm()
		self.portal = portal.Portal(self.realm)
		self.db = TxMongoDatabase("ncb")
		self.checker = DBCredentialsChecker(self.db)
		self.portal.registerChecker(self.checker)

		self.options = {
		        "login" : None, # should send models
				"launchSim" : self.launch_sim,
				"saveModel" : self.save_model,
				"getModels": self.get_models,
				"undoModelChange": self.undo_model_change,
				"logout": None # should close connection
			}

	def startService(self):
		service.Service.startService(self)
		log.msg('Authentication service has been started')

	def launch_sim(self, params):
		sim = Simulation()
		deferred = maybeDeferred(sim.build_sim, params, self.username)
		deferred.addCallback(sim.run_sim, params)

		# TEMPORARY..
		self.response = 'Successful sim launch.'
		# TODO: add errback for failed sim launch

		return deferred

		# Should automatically stream sim data out on same connection
		# Should also be able to send request to receive current streaming data, queued data that hasn't been viewed, old data?

	# This checks if creating new model or updating an existing
	def save_model(self, params):

		# check if this model exists by querying with fields author, model name, and location
		location = params['location']
		author_field = {}
		name_field = {}
		author_field['model_rev_1.model.author'] = params['model']['author']
		name_field['model_rev_1.model.name'] = params['model']['name']

		collection = None
		if location == 'personal':
			collection = self.username + '_models'
		elif location == 'lab':
			lab_num = self.realm.avatar_id.split(':')[1]
			collection = 'lab' + lab_num + '_models'
		elif location == 'global':
			collection = 'global_models'
		else:
			self.response = 'Failed model save. Invalid location.'
			return defer.fail(Exception('Invalid model location'))

		deferred = self.db.query_by_field(collection, {'$and': [author_field, name_field]})
		deferred.addCallback(self.insert_model, params = params, collection = collection)
		self.response = 'Successful model save.'
		return deferred


	def insert_model(self, query_result, params, collection):

		if query_result:

			doc_id = query_result[0].get('_id')

			# save a backup of the current model
			updated_model = query_result[0].copy()
			updated_model["model_rev_0"] = updated_model["model_rev_1"].copy()

			# update model
			updated_model["model_rev_1"]["model"] = params['model'].copy()
			updated_model['model_rev_1']['last_modified'] = time.strftime("%m/%d/%Y")
			self.db.update_doc(collection, {'_id': doc_id}, updated_model)

		else:
			new_model = {
				"model_rev_0": {
					"date_created": time.strftime("%m/%d/%Y"),
					"last_modified": time.strftime("%m/%d/%Y"),
					"model": params['model'].copy()
				},
				"model_rev_1": {
					"date_created": time.strftime("%m/%d/%Y"),
					"last_modified": time.strftime("%m/%d/%Y"),
					"model": params['model'].copy()
				}
			}

			# insert model into appropriate database collection
			self.db.insert(collection, new_model)

	def get_models(self, params):

		# send rev1 models
		models = {
			"personal": [],
			"lab": [],
			"global": []
		}

		# get model collections
		personal_col = self.username + '_models'
		lab_col = 'lab' + self.realm.avatar_id.split(':')[1] + '_models'
		global_col = 'global_models'

		deferred = self.db.query(personal_col)
		deferred.addCallback(self.get_model_query_results, models = models, group = 'personal')
		deferred = self.db.query(lab_col)
		deferred.addCallback(self.get_model_query_results, models = models, group = 'lab')
		deferred = self.db.query(global_col)
		deferred.addCallback(self.get_model_query_results, models = models, group = 'global')

		return deferred

	def get_model_query_results(self, query_results, models, group):
		if query_results:
			for result in query_results:
				models[group].append(result.get('model_rev_1'))

		if group == 'global':
			self.response = json.dumps(models)

	def undo_model_change(self, params):

		# find the specified model with author, model name, and location
		location = params['location']
		author_field = {}
		name_field = {}
		author_field['model_rev_1.model.author'] = params['model']['author']
		name_field['model_rev_1.model.name'] = params['model']['name']

		collection = None
		if location == 'personal':
			collection = self.username + '_models'
		elif location == 'lab':
			lab_num = self.realm.avatar_id.split(':')[1]
			collection = 'lab' + lab_num + '_models'
		elif location == 'global':
			collection = 'global_models'
		else:
			self.response = 'Failed model revert. Invalid location.'
			return defer.fail(Exception('Invalid model location'))

		deferred = self.db.query_by_field(collection, {'$and': [author_field, name_field]})
		deferred.addCallback(self.revert_model, params = params, collection = collection)
		return deferred

	def revert_model(self, query_result, params, collection):

		if query_result:
			doc_id = query_result[0].get('_id')

			# revert model to backup
			reverted_model = query_result[0].copy()
			reverted_model["model_rev_1"] = reverted_model["model_rev_0"].copy()

			# update model
			self.db.update_doc(collection, {'_id': doc_id}, reverted_model)
			del reverted_model['_id']
			self.response = json.dumps(reverted_model)

		else:
			self.response = 'Failed model revert. Model not in database.'
			return defer.fail(Exception('Model not in database'))