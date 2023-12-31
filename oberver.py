# import time module, Observer, FileSystemEventHandler
import time
from watchdog.observers import Observer
from watchdog.events import FileSystemEventHandler


class DirectoryWatcher:
    def __init__(self, watch_dir):
        self.watch_dir = watch_dir
        self.observer = Observer()

    def run(self):
        event_handler = Handler()
        self.observer.schedule(event_handler, self.watch_dir, recursive=True)
        self.observer.start()
        try:
            while True:
                time.sleep(5)
        except:
            self.observer.stop()
            print("Observer Stopped")

        self.observer.join()


class Handler(FileSystemEventHandler):

    @staticmethod
    def on_any_event(event):
        if event.is_directory:
            if event.event_type == 'created':
                # Event is created, you can process it now
                print("Watchdog received created event - % s." % event.src_path)
            elif event.event_type == 'modified':
                # Event is modified, you can process it now
                print("Watchdog received modified event - % s." % event.src_path)
            elif event.event_type == 'deleted':
                # Event is modified, you can process it now
                print("Watchdog received deleted event - % s." % event.src_path)

        if not event.is_directory:
            if event.event_type == 'created':
                # Event is created, you can process it now
                print("Watchdog received created event - % s." % event.src_path)
            elif event.event_type == 'modified':
                # Event is modified, you can process it now
                print("Watchdog received modified event - % s." % event.src_path)


if __name__ == '__main__':
    watch = DirectoryWatcher("/Users/rikenshah/Documents/fabric database")
    watch.run()
