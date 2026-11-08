from dsh_vts_avatar.performance.diary import DiaryStore
from dsh_vts_avatar.performance.privacy import PrivacyStore, trust_required


def test_engine_private_add_list_share(tmp_path):
    from dsh_vts_avatar.bridge.engine import AvatarEngine
    from dsh_vts_avatar.config import Config, EmotionConfig

    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode="affect")
    engine = AvatarEngine(config)
    added = engine.private_action("add", content="我很怕他不来了", kind="feeling", privacy=0.2)
    assert added["ok"] is True
    listed = engine.private_action("list")
    assert len(listed["items"]) == 1
    shared = engine.private_action("share", item_id=added["id"])
    assert shared["ok"] is True
    assert engine.private_action("list")["items"] == []


def test_engine_recall_returns_diary_and_episodes(tmp_path):
    from dsh_vts_avatar.bridge.engine import AvatarEngine
    from dsh_vts_avatar.config import Config, EmotionConfig

    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode="affect")
    engine = AvatarEngine(config)
    engine.diary.write("# 日记\n- 他说你根本不在乎", now=1000.0)
    result = engine.recall(query="不在乎")
    assert result["ok"] is True
    assert result["diary"]


def test_privacy_trust_threshold_and_share(tmp_path):
    store = PrivacyStore(tmp_path, "u1")
    item = store.add("feeling", "我其实很怕他哪天不来了", privacy=0.8)
    assert item.required == trust_required(0.8)
    assert not store.can_share(item, 0.3)
    assert store.can_share(item, 0.9)
    assert len(store.unshared()) == 1
    store.mark_shared(item.id)
    assert store.unshared() == []


def test_privacy_survives_reload(tmp_path):
    store = PrivacyStore(tmp_path, "u1")
    store.add("memory", "他说过会回来", privacy=0.4)
    reloaded = PrivacyStore(tmp_path, "u1")
    assert len(reloaded.unshared()) == 1


def test_diary_due_and_write(tmp_path):
    store = DiaryStore(tmp_path, "u1", max_bytes=10**6)
    assert store.due(now=1000.0) is True
    path = store.write("今天：聊了很久\n感受：平静", now=1000.0)
    assert path is not None and path.exists()
    assert store.due(now=1000.0 + 60) is False


def test_diary_cap_drops_oldest(tmp_path):
    store = DiaryStore(tmp_path, "u1", max_bytes=200)
    store.write("a" * 150, now=1000.0)
    store.write("b" * 150, now=1000.0 + 86400)
    files = list(store.root.glob("*.md"))
    assert len(files) == 1
    assert files[0].read_text(encoding="utf-8").startswith("b")

def test_mood_payload_includes_recent_diary(tmp_path):
    from dsh_vts_avatar.bridge.engine import AvatarEngine
    from dsh_vts_avatar.config import Config, EmotionConfig

    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode="affect")
    engine = AvatarEngine(config)
    engine.diary.write("# 日记\n- 他说你根本不在乎", now=1000.0)
    payload = engine.mood_payload()
    assert payload["diary"]
    assert "不在乎" in payload["text"]

def test_engine_relationship_event_and_trust_gated_share(tmp_path):
    from dsh_vts_avatar.bridge.engine import AvatarEngine
    from dsh_vts_avatar.config import Config, EmotionConfig

    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode="affect")
    engine = AvatarEngine(config)
    assert engine.relationship_event("promise_kept")["ok"] is True
    assert engine.relationship_event("not_an_event")["ok"] is False
    added = engine.private_action("add", content="我只跟很信任的人说", privacy=1.0)
    gated = engine.private_action("share", item_id=added["id"])
    assert gated["ok"] is False
    assert "trust" in gated["error"]

def test_engine_switch_user_isolates_stores(tmp_path):
    from dsh_vts_avatar.bridge.engine import AvatarEngine
    from dsh_vts_avatar.config import Config, EmotionConfig

    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode="affect")
    engine = AvatarEngine(config)
    engine.private_action("add", content="给 u2 的私事", privacy=0.2)
    assert engine.private_action("list")["items"]
    assert engine.switch_user("u2", "小鹿")["ok"] is True
    assert engine.private_action("list")["items"] == []
    engine.private_action("add", content="u2 自己的", privacy=0.2)
    assert len(engine.private_action("list")["items"]) == 1
    assert engine.switch_user("default")["ok"] is True
    assert len(engine.private_action("list")["items"]) == 1

def test_mood_prompt_names_the_active_user(tmp_path):
    from dsh_vts_avatar.bridge.engine import AvatarEngine
    from dsh_vts_avatar.config import Config, EmotionConfig

    config = Config(root=tmp_path)
    config.emotion = EmotionConfig(mode="affect")
    engine = AvatarEngine(config)
    assert "当前用户" not in engine.mood_payload()["text"]
    engine.switch_user("u2", "小鹿")
    assert "小鹿" in engine.mood_payload()["text"]
