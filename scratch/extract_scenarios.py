import os
import shutil
import negmas
from concurrent.futures import ProcessPoolExecutor

import os
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Manager, Process
from typing import Callable, Iterable, Dict, Any


class ProcessKillingExecutor:
    """
    The ProcessKillingExecutor works like an `Executor <https://docs.python.org/dev/library/concurrent.futures.html#executor-objects>`_
    in that it uses a bunch of processes to execute calls to a function with different arguments asynchronously.

    But other than the `ProcessPoolExecutor <https://docs.python.org/dev/library/concurrent.futures.html#concurrent.futures.ProcessPoolExecutor>`_,
    the ProcessKillingExecutor forks a new Process for each function call that terminates after the function returns or
    if a timeout occurs.

    This means that contrary to the Executors and similar classes provided by the Python Standard Library, you can
    rely on the fact that a process will get killed if a timeout occurs and that absolutely no side can occur between
    function calls.

    Note that descendant processes of each process will not be terminated – they will simply become orphaned.
    """

    def __init__(self, max_workers: int = None):
        """
        Initializes a new ProcessKillingExecutor instance.
        :param max_workers: The maximum number of processes that can be used to execute the given calls.
        """
        super().__init__()
        self.processes = max_workers or os.cpu_count()
        self.manager = Manager()

    def map(self, func: Callable, iterable: Iterable, timeout: float = None, callback_timeout: Callable = None,
            daemon: bool = True):
        """
        Returns an iterator (actually, a generator) equivalent to map(fn, iter).
        :param func: the function to execute
        :param iterable: an iterable of function arguments
        :param timeout: after this time, the process executing the function will be killed if it did not finish
        :param callback_timeout: this function will be called, if the task times out. It gets the same arguments as
                                 the original function
        :param daemon: run the child process as daemon
        :return: An iterator equivalent to: map(func, *iterables) but the calls may be evaluated out-of-order.
        """
        executor = ThreadPoolExecutor(max_workers=self.processes)
        params = ({'func': func, 'args': args, 'timeout': timeout, 'callback_timeout': callback_timeout,
                   'daemon': daemon} for args in iterable)
        return executor.map(self._submit_unpack_kwargs, params)

    def _submit_unpack_kwargs(self, kwargs: Dict):
        """unpack the kwargs and call submit"""
        return self.submit(**kwargs)

    def submit(self, func: Callable = None, args: Any = (), kwargs: Dict = {}, timeout: float = None,
               callback_timeout: Callable[[Any], Any] = None, daemon: bool = True):
        """
        Submits a callable to be executed with the given arguments.
        Schedules the callable to be executed as func(*args, **kwargs) in a new process.
        Returns the result, if the process finished successfully, or None, if it fails or a timeout occurs.
        :param func: the function to execute
        :param args: the arguments to pass to the function. Can be one argument or a tuple of multiple args.
        :param kwargs: the kwargs to pass to the function
        :param timeout: after this time, the process executing the function will be killed if it did not finish
        :param callback_timeout: this function will be called with the same arguments, if the task times out.
        :param daemon: run the child process as daemon
        :return: the result of the function, or None if the process failed or timed out
        """
        args = args if isinstance(args, tuple) else (args,)
        shared_dict = self.manager.dict()
        process_kwargs = {'func': func, 'args': args, 'kwargs': kwargs, 'share': shared_dict}
        p = Process(target=self._process_run, kwargs=process_kwargs, daemon=daemon)
        p.start()
        p.join(timeout=timeout)
        if 'return' in shared_dict:
            return shared_dict['return']
        else:
            if callback_timeout:
                callback_timeout(*args, **kwargs)
            if p.is_alive():
                p.terminate()
            return None

    @staticmethod
    def _process_run(func: Callable[[Any], Any] = None, args: Any = (), kwargs: Dict = {}, share: Dict = None):
        """
        Executes the specified function as func(*args, **kwargs).
        The result will be stored in the shared dictionary
        :param func: the function to execute
        :param args: the arguments to pass to the function
        :param kwargs: the kwargs to pass to the function
        :param share: a dictionary created using Manager.dict()
        """
        result = func(*args, **kwargs)
        share['return'] = result


def timeout(fnc, *args, seconds: int=5, **kwargs):
    with ProcessPoolExecutor() as p:
        f = p.submit(fnc, *args, **kwargs)
        try:
            return f.result(timeout=seconds)
        except:
            return None


def get_outcomes(name):
    mechanism, agent_info, issues = negmas.load_genius_domain_from_folder(folder_name=name)
    return negmas.num_outcomes(issues)


seconds = 20
max_n_outcomes = 2000
dirs = os.listdir("/Users/yasser/code/projects/uneg/data/scenarios")
full_names = [f"/Users/yasser/code/projects/uneg/data/scenarios/{_}" for _ in dirs]
for dir, name in zip(dirs, full_names):
    if not os.path.isdir(name):
        continue
    if dir in ("S-1NIKFRT-3","S-1NIKFRT-2") or dir.startswith("50issues") or dir.startswith("30issues"):
        print(f"IGNORING {dir}")
        continue
    print(f"working on {dir}")
    # executor = ProcessKillingExecutor()
    # try:
    #     result = executor.submit(get_outcomes, args=(name,), timeout=seconds)
    #     outcomes = result.result(timeout=seconds+1)
    # except TimeoutError:
    #     print(f"\t{name} timedout")
    #     continue
    outcomes = timeout(get_outcomes, name, seconds=seconds)
    if outcomes is None:
        print("\t Continuous outcome space ... ignored")
        continue
    if outcomes > max_n_outcomes:
        print(f"\t too many outcomes {outcomes} ... ignored")
        continue
    dst = f"/Users/yasser/code/projects/uneg/data/nd/{outcomes:06}{dir}"
    shutil.rmtree(dst, ignore_errors=True)
    shutil.copytree(name, dst)
