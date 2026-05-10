from homr.main import download_weights


def test_hash():
    assert not download_weights(pytest=True)[0]
