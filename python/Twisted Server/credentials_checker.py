from twisted.cred import error
from twisted.cred.checkers import ICredentialsChecker
from twisted.cred.credentials import IUsernameHashedPassword 
from twisted.internet.defer import Deferred
from zope.interface import implements

DEBUG = True

class DBCredentialsChecker(object):

	implements(ICredentialsChecker)

	# list here all the credentials that this checker is able to verify
	credentialInterfaces = (IUsernameHashedPassword,)

	def __init__(self, database): 
		self.database = database

	# The avatar ID is a token given to the user to access a particular avatar. 
	# This method checks the database for the credentials and returns a deferred containing the 
	# avatar ID if the user was found in the database
	def requestAvatarId(self, credentials):
		if DEBUG:
			print 'Requesting avatar ID'

		for interface in self.credentialInterfaces:
			if interface.providedBy(credentials): 
				break
			else:
				raise error.UnhandledCredentials()

		if DEBUG:
			print 'Attempting query'
		dbDeferred = self.database.query_by_field("users", {"username": credentials.username})
		deferred = Deferred()
		dbDeferred.addCallbacks(self._cbAuthenticate, self._ebAuthenticate, callbackArgs=(credentials, deferred), errbackArgs=(credentials, deferred))
		return deferred


	def _cbAuthenticate(self, result, credentials, deferred):
		if DEBUG:
			print 'Authenticating'

		# above query did not return any documents matching the provided username
		if not result:
			if DEBUG:
				print 'Invalid username'
			deferred.errback(error.UnauthorizedLogin('Invalid username')) 
		else:
			password = result[0].get('password')
			if DEBUG:
				print 'Correct password: ' + password
				print 'Entered password: ' + credentials.hashed
			if credentials.checkPassword(password):
				if DEBUG:
					print 'Valid credentials'

				# this passes back <username>:<lab id> as the avatar ID
				deferred.callback(credentials.username + ':' + str(result[0].get('lab_id'))) 
				#deferred.callback(credentials.username)
			else:
				if DEBUG:
					print 'Incorrect password'
				deferred.errback(error.UnauthorizedLogin('Incorrect password')) 

	def _ebAuthenticate(self, failure, credentials, deferred):
		if DEBUG:
			print 'In authentication errorback'
		deferred.errback(error.LoginFailed(failure))