import time, sys, threading, random

lock = threading.Lock()

numbers = [0]

def counter():
    """Add next number onto list, with a random 
    sleep between reading and writing"""
    while numbers[-1] < 20:
        val = numbers[-1] + 1
        # time.sleep(.01*random.randint(1, 10))
        time.sleep(.01)
        numbers.append(val)

def counter_with_lock():
    with lock:
        counter()

def run(target):
    threads = []

    # Start threads
    for i in range(10):
        thread = threading.Thread(target=target)
        thread.start()
        threads.append(thread)

    # Wait for threads to finish
    for thread in threads:
        thread.join()

if __name__ == "__main__":
    arg = sys.argv[1]
    if arg == "counter":
        counter()
        print(numbers)
    elif arg == "threaded":
        run(counter)
        print(numbers)
    elif arg == "threaded-with-lock":
        run(counter_with_lock)
        print(numbers)
    else:
        print('"counter" / "threaded" / "threaded-with-lock" are only valid args')
