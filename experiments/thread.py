import sys, time, random
from threading import Thread, Event, Lock

a = 0
event = Event()
lock = Lock()
l = []


def event_demo():
    def thread1(threadname):
        while True:
            if event.is_set():
                print(a)
                event.clear()

    def thread2(threadname):
        global a
        while 1:
            a += 1
            event.set()
            time.sleep(1)

    thread1 = Thread(target=thread1, args=("Thread-1",))
    thread2 = Thread(target=thread2, args=("Thread-2",))

    thread1.start()
    thread2.start()


def no_lock_demo():

    def appender():
        global l
        while True:
            length = len(l)
            if length < 100:
                time.sleep(random.random() / 1000)
                l.append(length)

    
    threads = [Thread(target=appender) for i in range(10)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print(l == list(range(1_000)))
    print(l)

def lock_demo():

    def appender():
        global l
        while True:

            # Note: program halts if we return without releasing
            lock.acquire()
            length = len(l)
            if length < 100:
                time.sleep(random.random() / 1000)
                l.append(length)
            lock.release()
            if length >= 100:
                return
    
    threads = [Thread(target=appender) for i in range(10)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print(l == list(range(100)))
    print(l)

def lock_cm_demo():
    def appender():
        global l
        while True:
            # Note: releasing handle for us
            with lock:
                length = len(l)
                if length >= 100:
                    return
                time.sleep(random.random() / 1000)
                l.append(length)

    
    threads = [Thread(target=appender) for i in range(10)]

    for thread in threads:
        thread.start()

    for thread in threads:
        thread.join()

    print(l == list(range(100)))
    print(l)

if __name__ == "__main__":
    eval(f"{sys.argv[1]}()")
