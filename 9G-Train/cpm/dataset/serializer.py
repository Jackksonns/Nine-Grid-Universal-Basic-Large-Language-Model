import json
import pickle


class Serializer:
    def __init__(self) -> None:
        pass

    def serialize(self, obj) -> bytes:
        raise NotImplementedError()

    def deserialize(self, data: bytes):
        raise NotImplementedError()


class PickleSerializer(Serializer):
    def __init__(self) -> None:
        pass

    def serialize(self, obj) -> bytes:
        return pickle.dumps(obj)

    def deserialize(self, data: bytes):
        return pickle.loads(data)


class JsonSerializer(Serializer):
    def __init__(self) -> None:
        pass

    def serialize(self, obj) -> bytes:
        return json.dumps(obj, ensure_ascii=False).encode("utf-8")

    def deserialize(self, data: bytes):
        return json.loads(data.decode("utf-8"))


class RawSerializer(Serializer):
    def __init__(self) -> None:
        pass

    def serialize(self, obj) -> bytes:
        return obj

    def deserialize(self, data: bytes):
        return data
