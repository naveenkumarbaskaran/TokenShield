import time
from tokenshield.cache import ResponseCache, MemoryBackend, DiskBackend


def _msgs() -> list[dict]:
    return [{"role": "user", "content": "hello"}]


def test_cache_miss_returns_none():
    c = ResponseCache()
    assert c.get("gpt-4o", _msgs()) is None


def test_cache_hit_returns_response():
    c = ResponseCache()
    resp = {"text": "hi", "cost": 0.001}
    c.set("gpt-4o", _msgs(), resp)
    assert c.get("gpt-4o", _msgs()) == resp


def test_different_model_is_miss():
    c = ResponseCache()
    resp = {"text": "hi"}
    c.set("gpt-4o", _msgs(), resp)
    assert c.get("gpt-4o-mini", _msgs()) is None


def test_different_messages_is_miss():
    c = ResponseCache()
    c.set("gpt-4o", _msgs(), {"text": "hi"})
    other = [{"role": "user", "content": "bye"}]
    assert c.get("gpt-4o", other) is None


def test_ttl_expiry():
    c = ResponseCache(ttl_seconds=1)
    c.set("gpt-4o", _msgs(), {"text": "hi"})
    assert c.get("gpt-4o", _msgs()) is not None
    time.sleep(1.1)
    assert c.get("gpt-4o", _msgs()) is None


def test_ttl_zero_means_no_expiry():
    c = ResponseCache(ttl_seconds=0)
    c.set("gpt-4o", _msgs(), {"text": "hi"})
    assert c.get("gpt-4o", _msgs()) is not None


def test_disabled_cache_always_misses():
    c = ResponseCache(enabled=False)
    c.set("gpt-4o", _msgs(), {"text": "hi"})
    assert c.get("gpt-4o", _msgs()) is None


def test_disk_backend_persists(tmp_path):
    backend = DiskBackend(cache_dir=tmp_path)
    c = ResponseCache(backend=backend)
    c.set("gpt-4o", _msgs(), {"text": "persisted"})

    # New cache instance, same backend dir
    c2 = ResponseCache(backend=DiskBackend(cache_dir=tmp_path))
    assert c2.get("gpt-4o", _msgs()) == {"text": "persisted"}


def test_memory_backend_isolated():
    b1 = MemoryBackend()
    b2 = MemoryBackend()
    ResponseCache(backend=b1).set("gpt-4o", _msgs(), {"text": "a"})
    assert ResponseCache(backend=b2).get("gpt-4o", _msgs()) is None
