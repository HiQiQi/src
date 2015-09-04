"""Module containing the Train class and support functionality."""
__authors__ = "Ian Goodfellow"
__copyright__ = "Copyright 2010-2012, Universite de Montreal"
__credits__ = ["Ian Goodfellow"]
__license__ = "3-clause BSD"
__maintainer__ = "LISA Lab"
__email__ = "pylearn-dev@googlegroups"
from datetime import datetime
import os
import sys
import logging
import warnings
from pylearn2.utils import serial
from pylearn2.utils.string_utils import preprocess
from pylearn2.monitor import Monitor
from pylearn2.space import NullSpace
from pylearn2.utils.timing import log_timing, total_seconds
from pylearn2.utils import sharedX
from pylearn2.train import Train
from myMonitor import myMonitor
log = logging.getLogger(__name__)


class myTrain(Train):


    def setup(self):
        """
        Sets up the main loop. This is also called at the start of the
        main loop, so you need only call it if you're using a driver
        script that replaces the main loop with something else.
        """
        self.model.monitor = myMonitor.get_monitor(self.model)
        self.model.monitor.time_budget_exceeded = False
        if self.algorithm is not None:
            self.algorithm.setup(model=self.model, dataset=self.dataset)
        self.setup_extensions()

        # Model.modify_updates is used by the training algorithm to
        # enforce constraints after each step of learning. Here we
        # make sure the constraints are enforced from the start.
        self.model.enforce_constraints()

    def main_loop(self, time_budget=None):
        """
        Repeatedly runs an epoch of the training algorithm, runs any
        epoch-level callbacks, and saves the model.

        Parameters
        ----------
        time_budget : int, optional
            The maximum number of seconds before interrupting
            training. Default is `None`, no time limit.
        """
        t0 = datetime.now()
        self.setup()
        if self.algorithm is None:
            self.run_callbacks_and_monitoring()
            while True:
                if self.exceeded_time_budget(t0, time_budget):
                    break

                rval = self.model.train_all(dataset=self.dataset)
                if rval is not None:
                    raise ValueError("Model.train_all should not return " +
                                     "anything. Use Model.continue_learning " +
                                     "to control whether learning continues.")
                self.model.monitor.report_epoch()
                extension_continue = self.run_callbacks_and_monitoring()
                freq = self.save_freq
                if freq > 0 and self.model.monitor.get_epochs_seen() % freq == 0:
                    self.save()
                continue_learning = (self.model.continue_learning() and
                                     extension_continue)
                assert continue_learning in [True, False, 0, 1]
                if not continue_learning:
                    break
        else:
            if not hasattr(self.model, 'monitor'):
                # TODO: is this really necessary? I just put this error here
                # to prevent an AttributeError later, but I think we could
                # rewrite to avoid the AttributeError
                raise RuntimeError("The algorithm is responsible for setting"
                                   " up the Monitor, but failed to.")
            # if len(self.model.monitor._datasets) > 0:
#                 # This monitoring channel keeps track of a shared variable,
#                 # which does not need inputs nor data.
#                 self.training_seconds.__doc__ = """\
# The number of seconds that were spent in actual training during the most
# recent epoch. This excludes seconds that were spent running callbacks for
# the extensions, computing monitoring channels, etc."""
#                 self.model.monitor.add_channel(
#                     name="training_seconds_this_epoch",
#                     ipt=None,
#                     val=self.training_seconds,
#                     data_specs=(NullSpace(), ''),
#                     dataset=self.model.monitor._datasets[0])
#                 self.total_seconds.__doc__ = """\
# The number of seconds that were spent on the entirety of processing for the
# previous epoch. This includes not only training but also the computation of
# the monitoring channels, running TrainExtension callbacks, etc. This value
# is reported for the *previous* epoch because the amount of time spent on
# monitoring for this epoch is not known until the monitoring channels have
# already been reported."""
#                 self.model.monitor.add_channel(
#                     name="total_seconds_last_epoch",
#                     ipt=None,
#                     val=self.total_seconds,
#                     data_specs=(NullSpace(), ''),
#                     dataset=self.model.monitor._datasets[0])
            self.run_callbacks_and_monitoring()

            while True:
                if self.exceeded_time_budget(t0, time_budget):
                    break

                with log_timing(log, None, level=logging.DEBUG,
                                callbacks=[self.total_seconds.set_value]):
                    with log_timing(
                            log, None, final_msg='Time this epoch:',
                            callbacks=[self.training_seconds.set_value]):
                        rval = self.algorithm.train(dataset=self.dataset)
                    if rval is not None:
                        raise ValueError("TrainingAlgorithm.train should not "
                                         "return anything. Use "
                                         "TrainingAlgorithm.continue_learning "
                                         "to control whether learning "
                                         "continues.")
                    self.model.monitor.report_epoch()
                    extension_continue = self.run_callbacks_and_monitoring()
                    if self.save_freq > 0 and \
                       self.model.monitor.get_epochs_seen() % self.save_freq == 0:
                        self.save()
                continue_learning = (
                    self.algorithm.continue_learning(self.model) and
                    extension_continue
                )
                assert continue_learning in [True, False, 0, 1]
                if not continue_learning:
                    break

        self.model.monitor.training_succeeded = True

        if self.save_freq > 0:
            self.save()

    def run_callbacks_and_monitoring(self):
        """
        Runs the monitor, then calls Extension.on_monitor for all extensions.

        Returns
        -------
        continue_learning : bool
            If `False`, signals that at least one train
            extension wants to stop learning.
        """
        self.model.monitor()
        continue_learning = True
        for extension in self.extensions:
            try:
                extension.on_monitor(self.model, self.dataset, self.algorithm)
            except TypeError:
                logging.warning('Failure during callback ' + str(extension))
                raise
            # We catch an exception here instead of relying on return
            # values for backward compatibility. Lots of extensions
            # exist that don't return anything, currently.
            except StopIteration:
                log.info("Extension requested training halt.")
                continue_learning = False
        return continue_learning

    def save(self):
        """Saves the model."""
        #TODO-- save state of training algorithm so training can be
        # resumed after a crash
        for extension in self.extensions:
            extension.on_save(self.model, self.dataset, self.algorithm)
        if self.save_path is not None:
            with log_timing(log, 'Saving to ' + self.save_path):
                if self.first_save and (not self.allow_overwrite) \
                   and os.path.exists(self.save_path):
                    # Every job overwrites its own output on the second save
                    # and every save thereafter. The "allow_overwrite" flag
                    # only pertains to overwriting the output of previous jobs.
                    raise IOError("Trying to overwrite file when not allowed.")
                try:
                    # Make sure that saving does not serialize the dataset
                    self.dataset._serialization_guard = SerializationGuard()
                    serial.save(self.save_path, self.model,
                                on_overwrite='backup')
                finally:
                    self.dataset._serialization_guard = None
            self.first_save = False


class SerializationGuard(object):
    """
    This class exists to make objects that cannot be serialized. It is used to
    make sure you don't accidentally put pointers to objects that should not
    be serialized, such as the dataset, into objects that Train automatically
    serializes, such as the Model.
    """

    def __getstate__(self):
        """
        This method is called when someone attempts to serialize the object.
        This method raises an exception to prevent the serialization from
        occurring.
        """
        raise IOError("You tried to serialize something that should not"
                      " be serialized.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.ERROR)
    log.error("You probably meant to run scripts/train.py")
    sys.exit(1)
