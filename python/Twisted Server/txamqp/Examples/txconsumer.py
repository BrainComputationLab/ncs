from twisted.internet.defer import inlineCallbacks
from twisted.internet import reactor
from twisted.internet.protocol import ClientCreator
from twisted.python import log

from txamqp.protocol import AMQClient
from txamqp.client import TwistedDelegate

import txamqp.spec

@inlineCallbacks
def gotConnection(conn, username, password):
    print "Connected to broker."
    yield conn.authenticate(username, password)

    print "Authenticated. Ready to receive messages"
    chan = yield conn.channel(1)
    yield chan.channel_open()

    yield chan.queue_declare(queue="data", durable=True, exclusive=False, auto_delete=False)
    yield chan.exchange_declare(exchange="datastream", type="direct", durable=True, auto_delete=False)

    yield chan.queue_bind(queue="data", exchange="datastream", routing_key="testuser2@gmail.com..Output1")

    yield chan.basic_consume(queue='data', no_ack=True, consumer_tag="testtag")

    queue = yield conn.queue("testtag")

    while True:
        msg = yield queue.get()
        print 'Received: ' + msg.content.body + ' from channel #' + str(chan.id)
        if msg.content.body == "STOP":
            break

    yield chan.basic_cancel("testtag")

    yield chan.channel_close()

    chan0 = yield conn.channel(0)

    yield chan0.connection_close()

    reactor.stop()


if __name__ == "__main__":

    spec = txamqp.spec.load("../amqp0-8.stripped.rabbitmq.xml")
    delegate = TwistedDelegate()

    d = ClientCreator(reactor, AMQClient, delegate=delegate, vhost='/', spec=spec).connectTCP("localhost", 5672)

    d.addCallback(gotConnection, "guest", "guest")

    def errorback(err):
        if reactor.running:
            log.err(err)
            reactor.stop()

    d.addErrback(errorback)

    reactor.run()
