import json
from model import ModelService

if __name__ == '__main__':

	modelService = ModelService()

	file = open("script_converted.txt", "w")

	# izhikevich
	#dict = modelService.script_to_JSON("samples/models/izh/parser_test.py")

	# ncs
	#dict = modelService.script_to_JSON("samples/models/lif/bursting.py")

	# hodgkin huxley
	#dict = modelService.script_to_JSON("samples/models/test/hh_neuron_test.py")

	# flat synapse
	#dict = modelService.script_to_JSON("samples/models/test/flat_test.py")

	# ncs synapse
	dict = modelService.script_to_JSON("samples/models/test/ncs_synapse_test.py")

	file.write(json.dumps(dict, sort_keys=True, indent=2) + '\n\n\n')

	file.close()