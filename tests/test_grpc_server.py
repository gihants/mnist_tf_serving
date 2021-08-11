import grpc_server

from grpc_server import __version__


def test_version():
    assert __version__ == '0.1.0'