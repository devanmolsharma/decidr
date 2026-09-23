from decidr import TokenCache


def test_get_returns_none_for_a_miss():
    cache = TokenCache(persist=False)
    assert cache.get("model", "opt") is None


def test_set_then_get_round_trips():
    cache = TokenCache(persist=False)
    cache.set("model", "opt", ["op", "t"])
    assert cache.get("model", "opt") == ["op", "t"]


def test_save_with_persist_false_never_touches_disk(tmp_path):
    path = tmp_path / "cache.json"
    cache = TokenCache(path=path, persist=False)
    cache.set("model", "opt", ["op", "t"])
    cache.save()
    assert not path.exists()


def test_save_and_reload_round_trips_through_disk(tmp_path):
    path = tmp_path / "cache.json"
    cache = TokenCache(path=path, persist=True)
    cache.set("model", "opt", ["op", "t"])
    cache.save()
    assert path.exists()

    reloaded = TokenCache(path=path, persist=True)
    assert reloaded.get("model", "opt") == ["op", "t"]


def test_save_is_a_no_op_when_nothing_changed(tmp_path):
    path = tmp_path / "cache.json"
    cache = TokenCache(path=path, persist=True)
    cache.save()  # never dirtied
    assert not path.exists()


def test_missing_or_corrupt_file_loads_as_empty(tmp_path):
    path = tmp_path / "cache.json"
    path.write_text("not json")
    cache = TokenCache(path=path, persist=True)
    assert cache.get("model", "opt") is None
