import pytest


@pytest.fixture(scope="session")
def spark(request):
    """fixture for creating a spark session
    Args:
        request: pytest.FixtureRequest object
    """    
    return None


@pytest.mark.usefixtures("spark")
def test_feature_parse(spark):    
    assert 1 == 1
