"""
AHRAS Message Bus
-----------------
Provides a unified publish/subscribe API that works in two modes:

  DEV_MODE=True  → in-process threading.Queue  (zero dependencies)
  DEV_MODE=False → Apache Kafka                (production)

Usage (identical in both modes):
    from pipeline.bus import get_producer, get_consumer

    producer = get_producer()
    producer.send("raw.telemetry", {"key": "value"})

    consumer = get_consumer("raw.telemetry", group_id="my-group")
    for msg in consumer:
        process(msg)
"""

import json
import queue
import threading
import logging
from typing import Iterator

log = logging.getLogger(__name__)

# ── Shared in-process topic registry (DEV only) ──────────────────────────────
_topics: dict[str, queue.Queue] = {}
_topics_lock = threading.Lock()


def _get_topic(name: str) -> queue.Queue:
    with _topics_lock:
        if name not in _topics:
            _topics[name] = queue.Queue(maxsize=50_000)
        return _topics[name]


# ─────────────────────────────────────────────────────────────────────────────
# Dev implementations
# ─────────────────────────────────────────────────────────────────────────────

class _DevProducer:
    """Mimics KafkaProducer.send() interface."""

    def send(self, topic: str, value: dict) -> None:
        q = _get_topic(topic)
        try:
            q.put_nowait(value)
        except queue.Full:
            log.warning(f"[BUS] Topic '{topic}' full — dropping oldest message")
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            q.put_nowait(value)

    def flush(self) -> None:
        pass   # no-op in dev


class _DevConsumer:
    """Mimics KafkaConsumer iteration interface."""

    def __init__(self, topic: str, timeout_ms: int = 200):
        self._q = _get_topic(topic)
        self._timeout = timeout_ms / 1000.0
        self._running = True

    def __iter__(self) -> Iterator[dict]:
        while self._running:
            try:
                yield self._q.get(timeout=self._timeout)
            except queue.Empty:
                continue

    def stop(self) -> None:
        self._running = False


# ─────────────────────────────────────────────────────────────────────────────
# Production implementations (Kafka)
# ─────────────────────────────────────────────────────────────────────────────

class _KafkaProducerWrapper:
    def __init__(self, bootstrap: str):
        from kafka import KafkaProducer as _KP
        self._p = _KP(
            bootstrap_servers=bootstrap,
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
            retries=5,
            acks="all",
        )

    def send(self, topic: str, value: dict) -> None:
        self._p.send(topic, value=value)

    def flush(self) -> None:
        self._p.flush()


class _KafkaConsumerWrapper:
    def __init__(self, topic: str, bootstrap: str, group_id: str):
        from kafka import KafkaConsumer as _KC
        self._c = _KC(
            topic,
            bootstrap_servers=bootstrap,
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            group_id=group_id,
            auto_offset_reset="earliest",
        )

    def __iter__(self) -> Iterator[dict]:
        for msg in self._c:
            yield msg.value

    def stop(self) -> None:
        self._c.close()


# ─────────────────────────────────────────────────────────────────────────────
# Public factory functions
# ─────────────────────────────────────────────────────────────────────────────

def get_producer():
    from config.settings import DEV_MODE, KAFKA_BOOTSTRAP
    if DEV_MODE:
        return _DevProducer()
    return _KafkaProducerWrapper(KAFKA_BOOTSTRAP)


def get_consumer(topic: str, group_id: str = "ahras-default"):
    from config.settings import DEV_MODE, KAFKA_BOOTSTRAP
    if DEV_MODE:
        return _DevConsumer(topic)
    return _KafkaConsumerWrapper(topic, KAFKA_BOOTSTRAP, group_id)
