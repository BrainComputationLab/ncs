NCS
===

Neocortical Simulator: An extensible neural simulator for heterogeneous clusters.


Notes
-----

Please visit the [NCS website] for more information.


[NCS website]: http://ncs.io/docs/installation/


[![Bitdeli Badge](https://d2weczhvl823v0.cloudfront.net/BrainComputationLab/ncs/trend.png)](https://bitdeli.com/free "Bitdeli Badge")

NCS Daemon Installation Instructions
------------------------------------

1. Don't even think about starting these instructions without first completing the NCS installation.
(Note: For now you must also do "git fetch" and then "git checkout daemon" in the ncs directory).

2. Install MongoDB (Ref: http://www.liquidweb.com/kb/how-to-install-mongodb-on-ubuntu-14-04/)

~~~~
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 --recv 7F0CEB10
~~~~
~~~~
echo 'deb http://downloads-distro.mongodb.org/repo/ubuntu-upstart dist 10gen' | sudo tee /etc/apt/sources.list.d/mongodb.list
~~~~
~~~~
sudo apt-get update
~~~~
~~~~
sudo apt-get install -y mongodb-org
~~~~
~~~~
sudo service mongod start
~~~~

3. Install TxMongo

~~~~
sudo apt-get install python-setuptools
~~~~
~~~~
cd ncs/python/Twisted\ Server/txmongo/
~~~~
~~~~
sudo python setup.py install
~~~~

4. Install RabbitMQ (Ref: https://www.rabbitmq.com/install-debian.html)

Add the following line to your /etc/apt/sources.list:
deb http://www.rabbitmq.com/debian/ testing main

Then run the following commands
~~~~
wget https://www.rabbitmq.com/rabbitmq-signing-key-public.asc
~~~~
~~~~
sudo apt-key add rabbitmq-signing-key-public.asc
~~~~
~~~~
sudo apt-get update
~~~~

RabbitMQ service should start automatically, but you can check the status by
~~~~
sudo rabbitmqctl status
~~~~

5. Install Txamqp
~~~~
sudo apt-get install python-txamqp
~~~~

6. Install Protobuf
~~~~
sudo apt-get install python-protobuf
~~~~

7. Install Bcrypt
~~~~
sudo apt-get install libffi-dev
~~~~
~~~~
sudo pip install bcrypt
~~~~

8. Run the Daemon
~~~~
cd ncs/python/Twisted\ Server
~~~~

The Daemon can be run as a background process with 
~~~~
twistd --python server_daemon.py
~~~~

And can be killed with
~~~~
kill $(cat twistd.pid) 
~~~~

This is annoying when developing, so to run the Daemon as a foreground process do
~~~~
twistd --nodaemon --python server_daemon.py
~~~~

Note: if ran as a foreground process, the logging will 
output to the terminal instead of a file.
