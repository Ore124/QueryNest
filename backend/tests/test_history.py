import json

from app.history import RedisChatHistoryStore


class FakePipeline:
    def __init__(self, client):
        self.client = client
        self.operations = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        return None

    def rpush(self, key, value):
        self.operations.append(("rpush", key, value))
        return self

    def ltrim(self, key, start, end):
        self.operations.append(("ltrim", key, start, end))
        return self

    def expire(self, key, seconds):
        self.operations.append(("expire", key, seconds))
        return self

    def execute(self):
        self.client.operations.extend(self.operations)


class FakeRedis:
    def __init__(self):
        self.operations = []
        self.values = []

    def pipeline(self):
        return FakePipeline(self)

    def lrange(self, key, start, end):
        return self.values[start:] if end == -1 else self.values[start : end + 1]

    def ping(self):
        return True


def test_redis_history_appends_trims_expires_and_loads_messages():
    client = FakeRedis()
    store = RedisChatHistoryStore(
        "redis://unused",
        key_prefix="rag:chat",
        ttl_seconds=60,
        max_messages=3,
        client=client,
    )

    store.append("session-1", "user", "问题")

    assert client.operations[0][:2] == ("rpush", "rag:chat:session-1")
    assert json.loads(client.operations[0][2]) == {"role": "user", "content": "问题"}
    assert client.operations[1] == ("ltrim", "rag:chat:session-1", -3, -1)
    assert client.operations[2] == ("expire", "rag:chat:session-1", 60)

    client.values = [
        json.dumps({"role": "user", "content": "一"}),
        json.dumps({"role": "assistant", "content": "二"}),
    ]
    assert store.load("session-1") == [
        {"role": "user", "content": "一"},
        {"role": "assistant", "content": "二"},
    ]
    assert store.ping() is True
