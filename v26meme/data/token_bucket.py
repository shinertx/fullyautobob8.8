import time

class TokenBucket:
    def __init__(self, rate_per_min: int, min_sleep_ms: int):
        self.capacity = max(1, int(rate_per_min))
        self.tokens = self.capacity
        self.fill_rate = self.capacity / 60.0
        self.timestamp = time.monotonic()
        self.min_sleep = max(0, int(min_sleep_ms)) / 1000.0

    def consume(self, n=1):
        while True:
            now = time.monotonic()
            delta = now - self.timestamp
            self.timestamp = now
            self.tokens = min(self.capacity, self.tokens + delta * self.fill_rate)
            if self.tokens >= n:
                self.tokens -= n
                break
            time.sleep(max(self.min_sleep, 0.05))
