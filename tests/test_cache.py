from __future__ import annotations

import asyncio

from distillery.providers.cache import CachingProvider, LLMCache


class _CountingProvider:
    default_model = "fake"

    def __init__(self, response: str = "hello") -> None:
        self.calls = 0
        self.response = response

    def generate(
        self,
        prompt: str,
        *,
        model: str | None = None,
        temperature: float = 0.7,
        max_tokens: int = 1024,
        system: str = "",
    ) -> str:
        self.calls += 1
        return self.response


def test_cache_round_trip(tmp_path):
    cache = LLMCache(tmp_path / "c.sqlite")
    cache.put(
        model="m", temperature=0.0, max_tokens=100,
        prompt="p", system="s", response="resp",
    )
    hit = cache.get(
        model="m", temperature=0.0, max_tokens=100, prompt="p", system="s",
    )
    assert hit == "resp"
    # Different key does not collide.
    miss = cache.get(
        model="m", temperature=0.0, max_tokens=100, prompt="p", system="other",
    )
    assert miss is None


def test_cache_key_normalizes_float():
    k1 = LLMCache._key(model="m", temperature=0.7, max_tokens=1, prompt="p", system="")
    k2 = LLMCache._key(model="m", temperature=0.70000, max_tokens=1, prompt="p", system="")
    assert k1 == k2


def test_caching_provider_caches_deterministic(tmp_path):
    inner = _CountingProvider("cached-response")
    cache = LLMCache(tmp_path / "c.sqlite")
    prov = CachingProvider(inner, cache, caching_threshold=0.0, default_model="m")

    a = prov.generate("hello", model="m", temperature=0.0, max_tokens=10)
    b = prov.generate("hello", model="m", temperature=0.0, max_tokens=10)
    assert a == b == "cached-response"
    assert inner.calls == 1  # second call hit the cache

    stats = cache.stats()
    assert stats["hits"] == 1
    assert stats["misses"] == 1
    assert stats["entries"] == 1


def test_caching_provider_skips_sampling(tmp_path):
    inner = _CountingProvider("sample")
    cache = LLMCache(tmp_path / "c.sqlite")
    prov = CachingProvider(inner, cache, caching_threshold=0.0, default_model="m")

    prov.generate("hello", model="m", temperature=0.7, max_tokens=10)
    prov.generate("hello", model="m", temperature=0.7, max_tokens=10)
    assert inner.calls == 2  # high-temp calls bypass cache entirely
    assert cache.stats()["entries"] == 0


def test_caching_provider_async_uses_cache(tmp_path):
    inner = _CountingProvider("async-cached")
    cache = LLMCache(tmp_path / "c.sqlite")
    prov = CachingProvider(inner, cache, caching_threshold=0.0, default_model="m")

    async def _go():
        a = await prov.agenerate("hi", model="m", temperature=0.0, max_tokens=10)
        b = await prov.agenerate("hi", model="m", temperature=0.0, max_tokens=10)
        return a, b

    a, b = asyncio.run(_go())
    assert a == b == "async-cached"
    assert inner.calls == 1
