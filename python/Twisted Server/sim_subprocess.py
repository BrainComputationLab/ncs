from twisted.internet import protocol, reactor

DEBUG = False

class SubProcessProtocol(protocol.ProcessProtocol): 

	def connectionMade(self):
		if DEBUG:
			print "connectionMade called" 
		#reactor.callLater(10, self.terminateProcess)

	def terminateProcess(self): 
		self.transport.signalProcess('TERM')

	def outReceived(self, data):
		if DEBUG:
			print "outReceived called with %d bytes of data:\n%s" % (len(data), data)

	def errReceived(self, data):
		if DEBUG:
			print "errReceived called with %d bytes of data:\n%s" % (len(data), data) 

	def inConnectionLost(self):
		if DEBUG:
			print "inConnectionLost called, stdin closed." 

	def outConnectionLost(self):
		if DEBUG:
			print "outConnectionLost called, stdout closed." 

	def errConnectionLost(self):
		if DEBUG:
			print "errConnectionLost called, stderr closed."

	def processExited(self, reason):
		if DEBUG:
			print "processExited called with status %d" % (reason.value.exitCode,)

	def processEnded(self, reason):
		if DEBUG:
			print "processEnded called with status %d" % (reason.value.exitCode,)
			print "All FDs are now closed, and the process has been reaped." 
		#reactor.stop()