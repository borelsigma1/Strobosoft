import pytest

from strobosoft.core.synthetic import ThrowScene, write_demo_video


@pytest.fixture(scope="session")
def scene() -> ThrowScene:
    return ThrowScene()


@pytest.fixture(scope="session")
def demo_video(tmp_path_factory, scene):
    return write_demo_video(tmp_path_factory.mktemp("video") / "wurf.mp4", scene)
