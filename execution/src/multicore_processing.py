import multiprocessing
import warnings
from concurrent.futures import ProcessPoolExecutor
from typing import (Any, Callable, Generic, Iterable, List, Optional, TypeVar,
                    Union)

# # Define type variables for better typing
# T = TypeVar('T')  # Input type
# R = TypeVar('R')  # Result type


# def process_in_parallel(
#     process_function: Callable[[T], R],
#     input_items: Iterable[T],
#     num_workers: Optional[int] = None,
#     chunk_size: int = 1,
#     show_progress: bool = False
# ) -> List[R]:
#     """
#     Execute a function in parallel across multiple CPU cores using ProcessPoolExecutor.

#     Args:
#         process_function: Function to execute in parallel
#         input_items: Collection of inputs to process
#         num_workers: Number of worker processes (defaults to CPU count if None)
#         chunk_size: Size of work chunks sent to each worker process
#         show_progress: Whether to display a progress bar

#     Returns:
#         List of results corresponding to each input item

#     Raises:
#         ImportError: If tqdm is not available when show_progress=True
#         ValueError: If invalid parameters are provided
#     """
#     if num_workers is None:
#         num_workers = multiprocessing.cpu_count()

#     if num_workers < 1:
#         raise ValueError("Number of workers must be at least 1")

#     if chunk_size < 1:
#         raise ValueError("Chunk size must be at least 1")

#     # Convert to list to handle generators and enable length measurement
#     input_list = list(input_items)

#     with ProcessPoolExecutor(max_workers=num_workers) as executor:
#         # Get the future mapping
#         future_results = executor.map(process_function, input_list, chunksize=chunk_size)
        
#         # Apply progress bar if requested
#         if show_progress:
#             try:
#                 from tqdm import tqdm
#                 results = list(tqdm(future_results, total=len(input_list), desc="Processing items"))
#             except ImportError:
#                 warnings.warn("tqdm not installed. Progress bar disabled.")
#                 results = list(future_results)
#         else:
#             results = list(future_results)

#     return results


# def process_with_joblib(process_function: Callable[[T], R],
#                         input_items: Iterable[T],
#                         num_workers: Optional[int] = None,
#                         backend: str = 'loky',
#                         verbosity: int = 0,
#                         show_progress: bool = False
#                         ) -> List[R]:
#     """
#     Execute a function in parallel using joblib's Parallel processing.

#     Args:
#         process_function: Function to execute in parallel
#         input_items: Collection of inputs to process
#         num_workers: Number of worker processes (defaults to CPU count if None)
#         backend: Parallelization backend ('loky', 'threading', 'multiprocessing')
#         verbosity: Joblib's verbosity level (0=silent, 1=progress bar, >1=more details)
#         show_progress: Whether to display a tqdm progress bar (overrides joblib's reporting)

#     Returns:
#         List of results corresponding to each input item

#     Raises:
#         ImportError: If joblib or tqdm (when show_progress=True) is not installed
#         ValueError: If invalid parameters are provided
#     """
#     try:
#         from joblib import Parallel, delayed
#     except ImportError:
#         raise ImportError(
#             "joblib is not installed. Please install with 'pip install joblib'")

#     if num_workers is None:
#         num_workers = multiprocessing.cpu_count()

#     if num_workers < 1:
#         raise ValueError("Number of workers must be at least 1")

#     valid_backends = ['loky', 'threading', 'multiprocessing']
#     if backend not in valid_backends:
#         raise ValueError(f"Backend must be one of {valid_backends}")

#     # Convert to list to handle generators and enable length measurement
#     input_list = list(input_items)
#     total_items = len(input_list)

#     if show_progress:
#         try:
#             from tqdm import tqdm
#             with tqdm(total=total_items, desc="Processing items") as progress_bar:
#                 def tracked_process(item):
#                     result = process_function(item)
#                     progress_bar.update(1)
#                     return result

#                 results = Parallel(n_jobs=num_workers, backend=backend, verbose=0)(
#                     delayed(tracked_process)(item) for item in input_list
#                 )
#         except ImportError:
#             warnings.warn(
#                 "tqdm not installed. Using joblib's progress reporting.")
#             results = Parallel(n_jobs=num_workers, backend=backend, verbose=verbosity)(
#                 delayed(process_function)(item) for item in input_list
#             )
#     else:
#         results = Parallel(n_jobs=num_workers, backend=backend, verbose=verbosity)(
#             delayed(process_function)(item) for item in input_list
#         )

#     return results

# import time
# def cpu_intensive_task(n: int) -> int:
#     """A sample CPU-intensive function that computes the sum of squares."""
#     time.sleep(0.1)  # Simulate computational work
#     return sum(i*i for i in range(n))

# if __name__ == "__main__":
#     # Test cases for both parallel processing functions

#     # Input data
#     test_data = list(range(100, 120))

#     # Test 1: ProcessPoolExecutor approach
#     print("\nTesting process_in_parallel:")
#     start_time = time.time()
#     results1 = process_in_parallel(
#         process_function=cpu_intensive_task,
#         input_items=test_data,
#         num_workers=4,
#         show_progress=True
#     )
#     print(f"Time taken: {time.time() - start_time:.2f} seconds")
#     print(f"First few results: {results1[:3]}")

#     # Test 2: Joblib approach
#     print("\nTesting process_with_joblib:")
#     start_time = time.time()
#     results2 = process_with_joblib(
#         process_function=cpu_intensive_task,
#         input_items=test_data,
#         num_workers=4,
#         backend='loky',
#         show_progress=True
#     )
#     print(f"Time taken: {time.time() - start_time:.2f} seconds")
#     print(f"First few results: {results2[:3]}")

#     # Verify results are the same
#     assert results1 == results2, "Results from the two methods differ!"
#     print("\nBoth methods produced identical results ✓")
