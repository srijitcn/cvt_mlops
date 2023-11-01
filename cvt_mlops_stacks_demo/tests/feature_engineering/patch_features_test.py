import pytest


@pytest.fixture(scope="session")
def spark(request):
    """fixture for creating a spark session
    Args:
        request: pytest.FixtureRequest object
    """
    spark = (
        SparkSession.builder.master("local[1]")
        .appName("pytest-pyspark-local-testing")
        .getOrCreate()
    )
    request.addfinalizer(lambda: spark.stop())

    return spark


@pytest.mark.usefixtures("spark")
def test_feature_parse(spark):    
    assert 1 == 1
