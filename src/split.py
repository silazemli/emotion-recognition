from pathlib import Path

def get_ravdess_split():
    data_dir = Path("data/audio_speech_actors_01-24")

    actor_dirs = sorted([p for p in data_dir.iterdir() if p.is_dir()])

    train_actors = actor_dirs[:20]
    test_actors = actor_dirs[20:]

    train_files = []
    test_files = []

    for actor in train_actors:
        train_files.extend(actor.rglob("*.wav"))

    for actor in test_actors:
        test_files.extend(actor.rglob("*.wav"))

    return sorted(train_files), sorted(test_files)