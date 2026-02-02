"""Enhanced futures API for MLXSmith SDK.

Provides thread-safe futures with callbacks, timeout handling, cancellation,
and progress tracking for async operations.
"""

from __future__ import annotations

import threading
import time
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, Generic, Iterable, Optional, TypeVar, Union

from ..llm.backend import DecodingConfig

T = TypeVar("T")


class APIFutureState:
    """Enumeration of future states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    CANCELLED = "cancelled"
    FAILED = "failed"


class APIFuture(Generic[T]):
    """Enhanced future with callbacks, timeout, cancellation, and progress tracking.
    
    This class wraps a standard concurrent.futures.Future and adds:
    - Promise-style callbacks (.then(), .catch(), .finally_())
    - Timeout handling
    - Cancellation support
    - Progress tracking for long operations
    - Thread-safe state management
    
    Example:
        >>> future = client.forward_backward(batch)
        >>> future.then(lambda result: print(f"Loss: {result.loss}"))
        >>>       .catch(lambda e: print(f"Error: {e}"))
        >>>       .finally_(lambda: print("Done"))
        >>> 
        >>> # With timeout
        >>> result = future.result(timeout=30.0)
        >>> 
        >>> # Check progress
        >>> print(f"Progress: {future.progress}%")
    """
    
    def __init__(self, future: Optional[Future] = None, operation_id: Optional[str] = None):
        """Initialize APIFuture.
        
        Args:
            future: The underlying concurrent.futures.Future (optional)
            operation_id: Unique identifier for this operation
        """
        self._future = future
        self._operation_id = operation_id or f"op-{id(self)}"
        self._state = APIFutureState.PENDING
        self._progress: float = 0.0
        self._progress_message: str = ""
        self._result: Optional[T] = None
        self._exception: Optional[BaseException] = None
        
        # Callbacks
        self._success_callbacks: list[Callable[[T], Any]] = []
        self._error_callbacks: list[Callable[[BaseException], Any]] = []
        self._finally_callbacks: list[Callable[[], Any]] = []
        self._progress_callbacks: list[Callable[[float, str], Any]] = []
        
        # Thread safety
        self._lock = threading.RLock()
        self._done_event = threading.Event()
        
        # If wrapped future provided, attach callbacks
        if future is not None:
            future.add_done_callback(self._on_future_done)
    
    def _on_future_done(self, future: Future) -> None:
        """Internal callback when underlying future completes."""
        with self._lock:
            try:
                if future.cancelled():
                    self._state = APIFutureState.CANCELLED
                    self._run_finally_callbacks()
                else:
                    self._result = future.result()
                    self._state = APIFutureState.COMPLETED
                    self._progress = 100.0
                    self._run_success_callbacks(self._result)
                    self._run_finally_callbacks()
            except BaseException as e:
                self._exception = e
                self._state = APIFutureState.FAILED
                self._run_error_callbacks(e)
                self._run_finally_callbacks()
            finally:
                self._done_event.set()
    
    def _run_success_callbacks(self, result: T) -> None:
        """Execute success callbacks."""
        for callback in self._success_callbacks:
            try:
                callback(result)
            except Exception:
                pass  # Callback errors should not propagate
    
    def _run_error_callbacks(self, exception: BaseException) -> None:
        """Execute error callbacks."""
        for callback in self._error_callbacks:
            try:
                callback(exception)
            except Exception:
                pass  # Callback errors should not propagate
    
    def _run_finally_callbacks(self) -> None:
        """Execute finally callbacks."""
        for callback in self._finally_callbacks:
            try:
                callback()
            except Exception:
                pass  # Callback errors should not propagate
    
    def _run_progress_callbacks(self, progress: float, message: str) -> None:
        """Execute progress callbacks."""
        for callback in self._progress_callbacks:
            try:
                callback(progress, message)
            except Exception:
                pass  # Callback errors should not propagate
    
    # ========================================================================
    # Promise-style callbacks
    # ========================================================================
    
    def then(self, callback: Callable[[T], Any]) -> APIFuture[T]:
        """Register a success callback.
        
        Args:
            callback: Function to call with the result when successful
            
        Returns:
            Self for chaining
        """
        with self._lock:
            if self._state == APIFutureState.COMPLETED and self._result is not None:
                # Already completed, run immediately
                try:
                    callback(self._result)
                except Exception:
                    pass
            else:
                self._success_callbacks.append(callback)
        return self
    
    def catch(self, callback: Callable[[BaseException], Any]) -> APIFuture[T]:
        """Register an error callback.
        
        Args:
            callback: Function to call with the exception when failed
            
        Returns:
            Self for chaining
        """
        with self._lock:
            if self._state == APIFutureState.FAILED and self._exception is not None:
                # Already failed, run immediately
                try:
                    callback(self._exception)
                except Exception:
                    pass
            else:
                self._error_callbacks.append(callback)
        return self
    
    def finally_(self, callback: Callable[[], Any]) -> APIFuture[T]:
        """Register a callback that runs on completion (success or failure).
        
        Args:
            callback: Function to call when future completes
            
        Returns:
            Self for chaining
        """
        with self._lock:
            if self._state in (APIFutureState.COMPLETED, APIFutureState.FAILED, APIFutureState.CANCELLED):
                # Already done, run immediately
                try:
                    callback()
                except Exception:
                    pass
            else:
                self._finally_callbacks.append(callback)
        return self
    
    def on_progress(self, callback: Callable[[float, str], Any]) -> APIFuture[T]:
        """Register a progress callback.
        
        Args:
            callback: Function to call with (progress_percent, message)
            
        Returns:
            Self for chaining
        """
        with self._lock:
            self._progress_callbacks.append(callback)
            # Call immediately with current progress
            if self._progress > 0:
                try:
                    callback(self._progress, self._progress_message)
                except Exception:
                    pass
        return self
    
    # ========================================================================
    # Progress tracking
    # ========================================================================
    
    def update_progress(self, progress: float, message: str = "") -> None:
        """Update progress (called by the operation).
        
        Args:
            progress: Progress percentage (0-100)
            message: Optional progress message
        """
        with self._lock:
            self._progress = max(0.0, min(100.0, progress))
            self._progress_message = message
            self._run_progress_callbacks(self._progress, message)
    
    @property
    def progress(self) -> float:
        """Get current progress percentage (0-100)."""
        with self._lock:
            return self._progress
    
    @property
    def progress_message(self) -> str:
        """Get current progress message."""
        with self._lock:
            return self._progress_message
    
    # ========================================================================
    # State and result access
    # ========================================================================
    
    @property
    def state(self) -> str:
        """Get current state string."""
        with self._lock:
            return self._state
    
    @property
    def operation_id(self) -> str:
        """Get operation identifier."""
        return self._operation_id
    
    @property
    def done(self) -> bool:
        """Check if future is done (completed, failed, or cancelled)."""
        with self._lock:
            return self._state in (APIFutureState.COMPLETED, APIFutureState.FAILED, APIFutureState.CANCELLED)
    
    @property
    def cancelled(self) -> bool:
        """Check if future was cancelled."""
        with self._lock:
            return self._state == APIFutureState.CANCELLED
    
    @property
    def failed(self) -> bool:
        """Check if future failed with an exception."""
        with self._lock:
            return self._state == APIFutureState.FAILED
    
    def result(self, timeout: Optional[float] = None) -> T:
        """Get the result, blocking if necessary.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            The result value
            
        Raises:
            TimeoutError: If timeout expires
            CancelledError: If future was cancelled
            Exception: If the operation failed
        """
        if self._future is not None:
            return self._future.result(timeout=timeout)
        
        # Wait on our done event
        if not self._done_event.wait(timeout=timeout):
            raise TimeoutError(f"Operation {self._operation_id} timed out after {timeout}s")
        
        with self._lock:
            if self._state == APIFutureState.CANCELLED:
                raise Exception(f"Operation {self._operation_id} was cancelled")
            if self._state == APIFutureState.FAILED and self._exception is not None:
                raise self._exception
            return self._result  # type: ignore
    
    def exception(self, timeout: Optional[float] = None) -> Optional[BaseException]:
        """Get the exception if one occurred.
        
        Args:
            timeout: Maximum time to wait in seconds
            
        Returns:
            The exception or None if successful
        """
        try:
            self.result(timeout=timeout)
            return None
        except Exception as e:
            return e
    
    # ========================================================================
    # Cancellation
    # ========================================================================
    
    def cancel(self) -> bool:
        """Attempt to cancel the future.
        
        Returns:
            True if cancellation was successful, False otherwise
        """
        with self._lock:
            if self._state in (APIFutureState.COMPLETED, APIFutureState.FAILED, APIFutureState.CANCELLED):
                return False
            
            self._state = APIFutureState.CANCELLED
            
            if self._future is not None:
                cancelled = self._future.cancel()
                if not cancelled:
                    # Could not cancel, revert state
                    self._state = APIFutureState.PENDING
                    return False
            
            self._done_event.set()
            self._run_finally_callbacks()
            return True
    
    def cancelled(self) -> bool:  # type: ignore
        """Check if the future was cancelled."""
        with self._lock:
            return self._state == APIFutureState.CANCELLED
    
    # ========================================================================
    # Async/await support
    # ========================================================================
    
    def __await__(self):
        """Support for async/await syntax.
        
        Example:
            >>> result = await future
        """
        import asyncio
        
        async def _await_result():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.result)
        
        return _await_result().__await__()
    
    def as_coroutine(self):
        """Return as an asyncio coroutine.
        
        Returns:
            A coroutine that resolves to the future's result
        """
        import asyncio
        
        async def _coro():
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(None, self.result)
        
        return _coro()


class SdkFuturePool:
    """Thread-pool wrapper for async SDK calls with enhanced futures."""

    def __init__(self, max_workers: int = 4):
        self._executor = ThreadPoolExecutor(max_workers=max_workers)

    def submit_sample(self, backend: Any, prompts: Iterable[str], decoding: DecodingConfig) -> APIFuture:
        from . import sample
        
        future = self._executor.submit(sample, backend, prompts, decoding)
        return APIFuture(future=future, operation_id=f"sample-{id(future)}")

    def submit_forward_backward(self, backend: Any, loss_fn) -> APIFuture:
        from . import forward_backward
        
        future = self._executor.submit(forward_backward, backend, loss_fn)
        return APIFuture(future=future, operation_id=f"fb-{id(future)}")

    def submit_optim_step(self, backend: Any, optimizer: Any, grads: Any) -> APIFuture:
        from . import optim_step
        
        future = self._executor.submit(optim_step, backend, optimizer, grads)
        return APIFuture(future=future, operation_id=f"optim-{id(future)}")

    def submit_create_optimizer(self, backend: Any, *, lr: float, weight_decay: float = 0.0) -> APIFuture:
        from . import create_optimizer
        
        future = self._executor.submit(create_optimizer, backend, lr=lr, weight_decay=weight_decay)
        return APIFuture(future=future, operation_id=f"opt-create-{id(future)}")
    
    def submit(self, fn: Callable[..., T], *args, **kwargs) -> APIFuture[T]:
        """Submit a generic function to the pool.
        
        Args:
            fn: Function to execute
            *args: Positional arguments
            **kwargs: Keyword arguments
            
        Returns:
            APIFuture wrapping the submitted task
        """
        future = self._executor.submit(fn, *args, **kwargs)
        return APIFuture(future=future, operation_id=f"task-{id(future)}")

    def shutdown(self, wait: bool = True):
        """Shutdown the thread pool.
        
        Args:
            wait: Whether to wait for pending tasks to complete
        """
        self._executor.shutdown(wait=wait)


# Convenience function for creating completed futures
def completed_future(result: T, operation_id: Optional[str] = None) -> APIFuture[T]:
    """Create an already-completed future with a result.
    
    Args:
        result: The result value
        operation_id: Optional operation identifier
        
    Returns:
        Completed APIFuture
    """
    future = APIFuture[T](operation_id=operation_id or f"completed-{id(result)}")
    future._state = APIFutureState.COMPLETED
    future._result = result
    future._progress = 100.0
    future._done_event.set()
    return future


def failed_future(exception: BaseException, operation_id: Optional[str] = None) -> APIFuture[Any]:
    """Create an already-failed future with an exception.
    
    Args:
        exception: The exception
        operation_id: Optional operation identifier
        
    Returns:
        Failed APIFuture
    """
    future = APIFuture[Any](operation_id=operation_id or f"failed-{id(exception)}")
    future._state = APIFutureState.FAILED
    future._exception = exception
    future._done_event.set()
    return future


def cancelled_future(operation_id: Optional[str] = None) -> APIFuture[Any]:
    """Create an already-cancelled future.
    
    Args:
        operation_id: Optional operation identifier
        
    Returns:
        Cancelled APIFuture
    """
    future = APIFuture[Any](operation_id=operation_id or f"cancelled-{id(object())}")
    future._state = APIFutureState.CANCELLED
    future._done_event.set()
    return future
