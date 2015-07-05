import os
import sys

def main():

	path = "usernamereportname"
	fifo = open(path, "r")
	for line in fifo:
	    print "Received: " + line,
	fifo.close()


if __name__ == '__main__':
    main()
