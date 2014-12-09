from __future__ import absolute_import, unicode_literals
import os
import gc
from threading import Thread
from datetime import datetime

#specify the location of ncs.py in ncs_lib_path
ncs_lib_path = ('../../../../')
sys.path.append(ncs_lib_path)
import ncs

SIM_DATA_DIRECTORY = '/var/ncs/sims/'


class SimThread(Thread):
    """Thread that contains the running simulation."""

    def __init__(self, helper, sim, max_steps):
        # call the superstructor for the Thread class, otherwise demons emerge
        super(SimThread, self).__init__()
        self.sim = sim
        self.helper = helper
        self.max_steps = max_steps
        self.step = 0
        self.helper.is_running = True

    def run(self):
        while self.step <= self.max_steps:
            # TODO Debugging
            print "Stepping"
            # Run the simulation
            self.sim.step(1)
            # increment stop counter
            self.step += 1
        print "Simulation Complete!"
        # Once it's done, change the helper's status
        self.helper.is_running = False
        del self.sim
        # TODO Figure out whats going on here, MPI_INIT is being called twice
        # for some reason
        # force the sim to be garbage collected
        gc.collect()


class SimulatorService(object):
    """This class handles interaction with the NCS simulator."""

    _instance = None
    sim_status = None
    is_running = False
    simulation = None
    most_recent_sim_info = None

    def __init__(self):
        if not os.path.exists(SIM_DATA_DIRECTORY):
            os.makedirs(SIM_DATA_DIRECTORY)

    def get_info(self):
        """Get information about the simulator and its state."""
        return {}

    def run(self, user, model):
        """Run a simulation."""
        # create a new sim object
        self.simulation = ncs.Simulation()
        # TODO move this out of here
        neuron_group_dict = {}
        # generate the sim stuff
        errors = ModelHelper.process_model(self.simulation, model, neuron_group_dict)
        #check for errors
        if len(errors):
            # do something here
            pass
        # try to init the simulation
        if not self.simulation.init([]):
            info = {
                "status": "error",
                "message": "Failed to initialize simulation"
            }
            return info
        # after the init, we can add stims and reports
        errors += ModelHelper.add_stims_and_reports(self.simulation, model, neuron_group_dict)
        # generate a new ID for the ism
        sim_id = Crypt.generate_sim_id()
        #create the directory for sim information like reports
        os.makedirs(SIM_DATA_DIRECTORY + '/' + sim_id)
        # get a timestamp
        now = datetime.now()
        # get a formatted string of the timestamp
        time_string = now.strftime("%d/%m/%Y %I:%M:%S %p %Z")
        # info object to be sent back to the user
        info = {
            "status": "running",
            "user": user,
            "started": time_string,
            "sim_id": sim_id
        }
        # meta object for the sim directory
        meta = {
            "user": user,
            "started": time_string,
            "sim_id": sim_id
        }
        # write the status info to the directory
        with open(SIM_DATA_DIRECTORY + '/' + sim_id + '/meta.json', 'w') as fil:
            fil.write(json.dumps(meta))
        # store the info as the most recent sim info
        self.most_recent_sim_info = info
        # create a new thread for the simulation
        sim_thread = SimThread(self, self.simulation, 5)
        # start running the simulation
        sim_thread.start()
        return info

    def stop(self):
        """Stop the simulation."""
        # if there was a simulation running, shut it down
        if self.simulation.shutdown():
            # set current status to stopped
            self.sim_status['status'] = 'idle'
            return self.sim_status
        # otherwise indicate that
        else:
            info = {
                "status": "error",
                "message": "No simulation was running"
            }
            return info

