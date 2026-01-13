import pickle
from multiprocessing import shared_memory
from concurrent.futures import ProcessPoolExecutor

class AsyncBatcher:
    """
    Wraps an iterator and prefetches results using ProcessPoolExecutor.
    
    This allows CPU-bound preprocessing (like GIS operations) to run in parallel
    processes while the main process handles GPU dispatch.
    
    Key design: Iteration happens in MAIN process, workers are stateless functions.
    
    Example:
        >>> # Share environment data across workers
        >>> shm_name, shm_size = AsyncBatcher.share_object(env_data)
        >>> 
        >>> # Setup async batching pipeline
        >>> batcher = AsyncBatcher(
        ...     input_data_iterator=args_generator(),
        ...     worker_func=preprocess_batch,
        ...     max_prefetch=2,
        ...     max_workers=4
        ... )
        >>> 
        >>> # Process results as they become ready
        >>> for result in batcher:
        ...     run_on_gpu(result)
        >>> 
        >>> # Cleanup shared memory
        >>> AsyncBatcher.cleanup_shared_object(shm_name)
    """
    def __init__(self, input_data_iterator, worker_func, max_prefetch=1, max_workers=2):
        """
        Args:
            input_data_iterator: Iterator yielding tuples of args for worker_func
            worker_func: Static function (must be picklable) that runs in worker process
            max_prefetch: Number of batches to keep in flight (1-2 recommended)
            max_workers: Number of worker processes
        """
        self.input_iter = iter(input_data_iterator)
        self.worker_func = worker_func
        self.max_prefetch = max_prefetch
        self.max_workers = max_workers
        self._futures_queue = []
        self._executor = None
    
    def __iter__(self):
        with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
            self._executor = executor
            
            # Prime the pump: submit initial batches
            for _ in range(self.max_prefetch):
                self._submit_next()
            
            # Main loop: yield results and keep submitting
            while self._futures_queue:
                # Pop oldest future (FIFO)
                future = self._futures_queue.pop(0)
                
                # Block until result is ready (CPU work happens here)
                result = future.result()
                
                # Immediately submit next batch
                self._submit_next()
                
                yield result
    
    def _submit_next(self):
        """Pull next args from iterator and submit to worker pool."""
        try:
            args = next(self.input_iter)
            fut = self._executor.submit(self.worker_func, *args)
            self._futures_queue.append(fut)
        except StopIteration:
            pass  # No more work
    
    @staticmethod
    def share_object(obj):
        """
        Serialize a Python object to shared memory as a byte buffer.
        
        This is a "RAM disk" approach: pickle once, share the bytes.
        Much faster than pickling over pipes for each worker process.
        
        Args:
            obj: The Python object to share (must be picklable)
            
        Returns:
            (shm_name, size): Shared memory identifier and buffer size
            
        Example:
            >>> env_data = load_environment()
            >>> shm_name, shm_size = AsyncBatcher.share_object(env_data)
            >>> # Pass shm_name and shm_size to worker processes
        """
        # Pickle to bytes
        obj_bytes = pickle.dumps(obj)
        
        # Create shared memory buffer
        shm = shared_memory.SharedMemory(create=True, size=len(obj_bytes))
        shm.buf[:len(obj_bytes)] = obj_bytes
        
        return shm.name, len(obj_bytes)
    
    @staticmethod
    def load_shared_object(shm_name, size):
        """
        Load an object from shared memory.
        
        Typically called inside a worker process.
        
        Args:
            shm_name: The name of the shared memory block
            size: The size of the data in bytes
            
        Returns:
            The unpickled Python object
            
        Example:
            >>> # Inside worker process
            >>> env_data = AsyncBatcher.load_shared_object(shm_name, shm_size)
        """
        shm = shared_memory.SharedMemory(name=shm_name)
        # Read bytes
        obj_bytes = bytes(shm.buf[:size])
        # Unpickle
        obj = pickle.loads(obj_bytes)
        # Close handle (does not unlink/destroy the underlying memory)
        shm.close()
        return obj
    
    @staticmethod
    def cleanup_shared_object(shm_name):
        """
        Unlink and destroy a shared memory block.
        
        Should be called by the parent process when done.
        
        Args:
            shm_name: The name of the shared memory block
            
        Example:
            >>> import atexit
            >>> atexit.register(AsyncBatcher.cleanup_shared_object, shm_name)
        """
        try:
            shm = shared_memory.SharedMemory(name=shm_name)
            shm.close()
            shm.unlink()
        except FileNotFoundError:
            pass  # Already cleaned up
